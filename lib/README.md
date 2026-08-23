# Standard-cell library

The flow characterises every design with the **Nangate 45 nm Open Cell
Library**.  That library is third party and its Liberty file carries its own
copyright header, so it is not redistributed in this repository.

Drop your own copy here:

```
lib/NangateOpenCellLibrary_typical.lib
```

It ships with, among others, [OpenROAD-flow-scripts]
(https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts) under
`flow/platforms/nangate45/lib/`.

On the first synthesis run `approxlg/liberty.py` parses the Liberty file and
writes the derived cache `lib/nangate45.json` (cell areas, leakage, pin
capacitances and Boolean functions) next to it.  Both files are ignored by
git.

A different library works too — point `--lib` (or `approxlg.synth.DEFAULT_LIB`)
at its Liberty file.  The absolute PPA numbers then differ from the ones in the
paper, but the flow itself is library-agnostic.
