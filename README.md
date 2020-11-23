# penrose
Penrose tiling generation with numpy, visualisation with Pillow and processing.py. Experiments and interactive stuff (hopefully).

![example](images/tmp361bfs4_smol.png)
![another-example](images/tmpymhb48jh_smol.png)

## Notes

- the naive bottom-up approach with only local rules fails quickly (as expected)
- now focusing on using pentagrid as oracle for placing the tiles

### RESOURCES

All of these build mostly on N. G. de Bruijn's work.

- [M. W. Reinsch - Lattice representations of Penrose tilings of the plane (1999)](https://arxiv.org/abs/math-ph/9911024)
- [David Austin - Penrose Tilings Tied up in Ribbons (2005)](http://www.ams.org/publicoutreach/feature-column/fcarc-ribbons)
- [S. Mowry & S. Shukla - Pentagrids and Penrose Tilings (2013)](https://web.williams.edu/Mathematics/sjmiller/public_html/hudson/HRUMC-Mowry&Shukla_Pentagrids%20and%20Penrose.pdf)
- [Andrejs Treibergs - Penrose Tiling (2020)](http://www.math.utah.edu/~treiberg/PenroseSlides.pdf)
