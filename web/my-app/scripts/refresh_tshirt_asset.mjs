import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sourceJsonPath = path.join(
  appRoot,
  "src/lib/assets/tshirt_cad_source.json",
);
const jsonPath = path.join(appRoot, "src/lib/assets/tshirt_cad.json");
const objPath = path.join(appRoot, "src/lib/assets/tshirt_cad.obj");
const targetVersion = 72;
const targetTemplate = "top-standard-tee";

function normalize(vec) {
  const len = Math.hypot(vec[0], vec[1], vec[2]) || 1;
  return [vec[0] / len, vec[1] / len, vec[2] / len];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function loadSourceMesh() {
  return JSON.parse(fs.readFileSync(sourceJsonPath, "utf8"));
}

function rewriteNormals(mesh) {
  const positions = mesh.positions;
  const accum = new Array(positions.length).fill(0);

  for (let i = 0; i < mesh.indices.length; i += 3) {
    const ia = mesh.indices[i] * 3;
    const ib = mesh.indices[i + 1] * 3;
    const ic = mesh.indices[i + 2] * 3;

    const a = [positions[ia], positions[ia + 1], positions[ia + 2]];
    const b = [positions[ib], positions[ib + 1], positions[ib + 2]];
    const c = [positions[ic], positions[ic + 1], positions[ic + 2]];

    const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const n = cross(ab, ac);

    accum[ia] += n[0];
    accum[ia + 1] += n[1];
    accum[ia + 2] += n[2];
    accum[ib] += n[0];
    accum[ib + 1] += n[1];
    accum[ib + 2] += n[2];
    accum[ic] += n[0];
    accum[ic + 1] += n[1];
    accum[ic + 2] += n[2];
  }

  const normals = [];
  for (let i = 0; i < accum.length; i += 3) {
    normals.push(...normalize([accum[i], accum[i + 1], accum[i + 2]]));
  }
  mesh.normals = normals;
}

function rewriteBounds(mesh) {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < mesh.positions.length; i += 3) {
    for (let axis = 0; axis < 3; axis += 1) {
      const value = mesh.positions[i + axis];
      if (value < min[axis]) min[axis] = value;
      if (value > max[axis]) max[axis] = value;
    }
  }
  mesh.bounds = {
    width: max[0] - min[0],
    height: max[1] - min[1],
    depth: max[2] - min[2],
  };
}

function normalizeMetadata(mesh) {
  mesh.version = targetVersion;
  mesh.template = targetTemplate;
  delete mesh.signature;
  delete mesh.params;
}

function writeJson(mesh) {
  fs.writeFileSync(jsonPath, JSON.stringify(mesh), "utf8");
}

function writeObj(mesh) {
  const lines = [
    "# Static CAD T-shirt mesh",
    "# Source: src/lib/assets/tshirt_cad.json",
    "",
  ];

  for (let i = 0; i < mesh.positions.length; i += 3) {
    lines.push(
      `v ${mesh.positions[i]} ${mesh.positions[i + 1]} ${mesh.positions[i + 2]}`,
    );
  }

  for (let i = 0; i < mesh.uvs.length; i += 2) {
    lines.push(`vt ${mesh.uvs[i]} ${mesh.uvs[i + 1]}`);
  }

  for (let i = 0; i < mesh.normals.length; i += 3) {
    lines.push(
      `vn ${mesh.normals[i]} ${mesh.normals[i + 1]} ${mesh.normals[i + 2]}`,
    );
  }

  for (let i = 0; i < mesh.indices.length; i += 3) {
    const a = mesh.indices[i] + 1;
    const b = mesh.indices[i + 1] + 1;
    const c = mesh.indices[i + 2] + 1;
    lines.push(`f ${a}/${a}/${a} ${b}/${b}/${b} ${c}/${c}/${c}`);
  }

  fs.writeFileSync(objPath, lines.join("\n") + "\n", "utf8");
}

function main() {
  const mesh = loadSourceMesh();
  normalizeMetadata(mesh);
  rewriteNormals(mesh);
  rewriteBounds(mesh);
  writeJson(mesh);
  writeObj(mesh);
}

main();
