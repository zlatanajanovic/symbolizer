# Dark dashboard result

- Expected: a fixed-rail research dashboard at 1440×1200 and a compact top-navigation, stacked dashboard at 390×844, with verified release content and destinations preserved.
- Observed: the editorial DOM and visual system were replaced by an application shell with a persistent desktop rail, overview stage, metadata modules, horizontal workflow, evaluation cards, terminal reproduction panel, and integrated release record. The mobile breakpoint converts the rail to a compact header and stacks every panel without horizontal overflow.
- Browser check: two viewport scenarios passed. Local resources returned successful responses; hash navigation, mobile menu state, repository activation, outbound destinations, console/page errors, and horizontal overflow were checked. Injected axe-core reported zero serious or critical violations at both viewports.
- Visual evidence inspected: `/tmp/symbolizer-dark-dashboard-desktop.png` (1440×1200) and `/tmp/symbolizer-dark-dashboard-mobile.png` (390×844). Screenshots remain outside Git.
- Safety/content check: approved manuscript images, factual copy, authors, year, reproduction commands, limitations, demo disclaimer, license, arXiv, DOI, demo, and repository target remain. No unsupported venue claim, reviewer residue, private path, or secret was introduced.
