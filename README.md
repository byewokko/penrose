# penrose
Penrose tiling generation with processing.py. Experiments and interactive stuff.

## Notes

- bottom-up approach (composing tiles randomly) needs more work
  - it needs edge markers instead of node markers
- let's try the pentagrid approach 
    - generate random pentagrid
    - calculate all intersections
    - find out which strips each of them lies in => 5 coordinates
    - use this as oracle for the compositional tiling method 
- pentagrid will probably fail because of floating point math
    - for tiling composition, use heapq
    - first resolve nodes with higher confidence (further away from closest neighbors)
    
## To watch out:

- no grid line can pass through the origin
- no more than two lines can intersect at any single point

### RESOURCES

- [Pentagrid and 5D space](http://www.ams.org/publicoutreach/feature-column/fcarc-ribbons)
- [Pentagrid formulas](https://web.williams.edu/Mathematics/sjmiller/public_html/hudson/HRUMC-Mowry&Shukla_Pentagrids%20and%20Penrose.pdf)
- [More Penrose resources](http://www.math.utah.edu/~treiberg/PenroseSlides.pdf)