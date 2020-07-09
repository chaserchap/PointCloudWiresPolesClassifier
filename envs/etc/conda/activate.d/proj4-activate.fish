#!/usr/bin/env fish

if set -q PROJ_LIB
  set -gx _CONDA_SET_PROJ_LIB "$PROJ_LIB"
end

if test -d "$CONDA_PREFIX/share/proj"
  set -gx PROJ_LIB "$CONDA_PREFIX/share/proj"
else if test -d "$CONDA_PREFIX/Library/share/proj"
  set -gx PROJ_LIB "$CONDA_PREFIX/Library/share/proj"
end
