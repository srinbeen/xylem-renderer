#ifndef XYLEM_MACROS_H
#define XYLEM_MACROS_H

namespace Xylem {
    #define XYLEM_USE_REVERSE_Z                    1
    #define XYLEM_IMPOSTOR_AZIMUTH_VIEWS           9
    #define XYLEM_IMPOSTOR_ELEVATION_VIEWS         9
    #define XYLEM_IMPOSTOR_VIEW_COUNT              (XYLEM_IMPOSTOR_AZIMUTH_VIEWS * XYLEM_IMPOSTOR_ELEVATION_VIEWS)
    #define XYLEM_NUM_CASCADES                     4
    #define XYLEM_TREE_AMBIENT                     0.18f
    #define XYLEM_SDSM_PADDING                     0.05f

}

#endif // XYLEM_MACROS_H
