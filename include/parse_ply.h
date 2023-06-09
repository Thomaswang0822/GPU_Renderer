#pragma once

#include "utils.h"
#include "matrix.h"
#include "parse_scene.h"
#include <filesystem>

#include "flexception.h"
#include "transform.h"

#include <fstream>

/// Parse Stanford PLY files.
ParsedTriangleMesh parse_ply(const fs::path &filename, const Matrix4x4 &to_world);
