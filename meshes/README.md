# SterileSpace: Anatomy Meshes

Place standard solid mesh files here (`.obj` or `.stl`). The Python data pipeline (using PyVista) parses them into point-cloud buffers at start-up.

```text
meshes/
  skull.stl          -> one single-part model ("skull")
  heart.obj          -> one single-part model ("heart")
  brain/             -> one multi-part model ("brain"), explodable
      cortex.stl
      cerebellum.stl
      brainstem.obj
```

- Each file inside a sub-folder is treated as one **part**. The open-hand "explode" gesture slides every part outward dynamically.
- A top-level file is a single part; for those, an open hand advances to the next model.
- Models are auto-centered and scaled to unit size by the pipeline, meaning any real-world coordinate scale works.
- If your STL/OBJ is Z-up and appears lying down on the WebGL canvas, adjust the rotation flags (e.g., `--base-pitch -90` or `90`).

*Note: No proprietary anatomy files are bundled in this repository. With this folder empty, the application falls back to a clearly labelled procedural DEMO model.*
