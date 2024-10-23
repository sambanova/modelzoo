#! /bin/bash
set -e

# Copyright 2024 SambaNova Systems, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


APP_ROOT="$(dirname "$(realpath "$0")")"
echo "APP_ROOT: $APP_ROOT"
# Host runtime library path
SITE_PACKAGES=/opt/sambanova/lib/python3.8/site-packages

runner=""
image=""
instance_name="devbox_${USER}_$(date +%s)"
cpu_only=false

# Common bind mounts for both Podman and Singularity
mounts=(
    "$APP_ROOT:/opt/modelzoo"
    "/tmp:/tmp"
)

# Runtime related bind mounts
# Mounted if cpu_only=false
runtime_mounts=(
    "/dev/hugepages:/dev/hugepages"
    "/opt/sambaflow/pef/:/opt/sambaflow/pef/"
    "/opt/sambaflow/runtime/:/opt/sambaflow/runtime/"
    "/var/lib/sambaflow/ccl/ccl_config.db:/var/lib/sambaflow/ccl/ccl_config.db"
    "/var/snml.sock:/var/snml.sock"
    "$SITE_PACKAGES/pysnml:$SITE_PACKAGES/pysnml"
    "$SITE_PACKAGES/pysnrdureset:$SITE_PACKAGES/pysnrdureset"
    "$SITE_PACKAGES/pysnrdutools:$SITE_PACKAGES/pysnrdutools"
    "$SITE_PACKAGES/sambaruntime:$SITE_PACKAGES/sambaruntime"
)

for ce in podman singularity
do
    if which $ce &> /dev/null; then
        export runner=$ce
        break
    fi
done

if [ -z "${runner}" ]; then
    echo "Please install podman or singularity!"
    exit 1
fi

usage() {
  cat <<USAGE; exit
$@
Usage $0 [-b bind_mount_path...] [-c command] [-s] [-u] image

Run a Devbox container. Options:

  -b: Path to artifacts to be mounted in the container. Can be used multiple times.
    Examples of artifacts may include checkpoints, datasets.

  -c: Command to run inside the container. If none is provided, an interactive shell is started.

  -s: Provide a name to your instance.

  -u: Run in CPU-only mode. This is useful if you don't have access to a RDU machine.

USAGE

}

# Parse shell arguments
while getopts ":b:c:s:u" opt; do
  case $opt in
    b)
      mounts+=("$OPTARG")
      ;;
    c)
      cmd_string="$OPTARG"
      ;;
    s)
      instance_name="$OPTARG"
      ;;
    u)
      cpu_only=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      usage
      exit 1
      ;;
  esac
done

# Check if image is provided
shift $((OPTIND - 1))
if [ $# -eq 0 ]; then
  usage
  exit 1
fi

image="${@: -1}"

echo -e "\033[0;32mUsing $runner with image $image\033[0m\n"

if [ "$cpu_only" = false ]; then
    mounts+=("${runtime_mounts[@]}")
fi

# Run container with mounts
if [ "$runner" = "podman" ]; then
    echo -e "Running podman container with name: $instance_name"

    # Start podman container in detached mode
    start_cmd="podman run -dt --privileged --name $instance_name"
    for mount in "${mounts[@]}"; do
        start_cmd+=" -v ${mount}"
    done
    start_cmd+=" $image"

    if podman container list --format "{{.Names}}" | grep -q -w "$instance_name"; then
      echo "Instance '$instance_name' already exists. Skipping start."
    else
      eval $start_cmd
      echo -e "Podman instance $instance_name started"
    fi

    # Run the podman instance
    run_cmd="podman exec -it $instance_name /bin/bash"
else
    echo -e "Running singularity instance with name: $instance_name"
    # Start singularity instance
    start_cmd="singularity instance start --writable-tmpfs";
    for mount in "${mounts[@]}"; do
        start_cmd+=" --bind ${mount}"
    done
    start_cmd+=" $image $instance_name"

    echo -e "Singularity start command: $start_cmd"
    if singularity instance list | grep -q -w "$instance_name"; then
      echo "Instance '$instance_name' already exists. Skipping start."
    else
      eval $start_cmd
      echo -e "Singularity instance $instance_name started"
    fi

    # Run the singularity instance
    run_cmd="singularity exec instance://$instance_name /bin/bash"
fi

if [ -n "$cmd_string" ]; then
    run_cmd+=" -c \"$cmd_string\""
fi

echo -e "Run command: $run_cmd"

eval $run_cmd
