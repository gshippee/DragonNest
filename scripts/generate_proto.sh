#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python -m grpc_tools.protoc \
  -I proto \
  --python_out=src/dragon_nest/proto \
  --grpc_python_out=src/dragon_nest/proto \
  --pyi_out=src/dragon_nest/proto \
  proto/dragonnest.proto

# grpc_tools generates a top-level import even though bindings live in a package.
sed -i 's/^import dragonnest_pb2 as dragonnest__pb2$/from . import dragonnest_pb2 as dragonnest__pb2/' \
  src/dragon_nest/proto/dragonnest_pb2_grpc.py
