// Zero-dependency static server for WonderSnap. The camera API needs a secure context: http://localhost is one.
//   node server.mjs [port]      ->  http://localhost:5173
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)));
const MESHES = resolve(ROOT, '..', 'meshes');   // repo-level meshes/ folder, exposed read-only at /meshes/
const PORT = Number(process.argv[2] || process.env.PORT || 5173);
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.wasm': 'application/wasm', '.task': 'application/octet-stream',
  '.png': 'image/png', '.svg': 'image/svg+xml', '.ico': 'image/x-icon', '.map': 'application/json',
  '.obj': 'text/plain; charset=utf-8', '.stl': 'application/octet-stream',
};

createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');
    const rel = decodeURIComponent(url.pathname);
    const isMesh = rel.startsWith('/meshes/');
    const base = isMesh ? MESHES : ROOT;
    let path = normalize(join(base, isMesh ? rel.slice('/meshes'.length) : rel));
    if (!path.startsWith(base)) { res.writeHead(403).end('forbidden'); return; }
    if ((await stat(path).catch(() => null))?.isDirectory()) path = join(path, 'index.html');
    const body = await readFile(path);
    res.writeHead(200, { 'Content-Type': TYPES[extname(path).toLowerCase()] || 'application/octet-stream', 'Cache-Control': 'no-cache' });
    res.end(body);
  } catch {
    res.writeHead(404, { 'Content-Type': 'text/plain' }).end('not found');
  }
}).listen(PORT, () => console.log(`WonderSnap running at http://localhost:${PORT}`));
