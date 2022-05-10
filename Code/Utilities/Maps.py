def Risk():
    # risk_matrix = {
    #     "L"  : 0.95,
    #     "ML" : 0.90,
    #     "M"  : 0.87,
    #     "MH" : 0.85,
    #     "HL" : 0.80,
    #     "HM" : 0.75,
    #     "H"  : 0.65,
    #     "HH" : 0.50,
    #     "VH" : 0.40, 
    # }
    
    risk_matrix = {
        "L"  : 0.999,
        "ML" : 0.990,
        "M"  : 0.985,
        "MH" : 0.970,
        "HL" : 0.955,
        "HM" : 0.950,
        "H"  : 0.945,
        "HH" : 0.940,
        "VH" : 0.920, 
    }

    return risk_matrix


def LivingArea(risk_matrix):
    connections = [
        [ 1,  2, 0.50, risk_matrix["L"]],
        [ 1,  6, 0.75, risk_matrix["L"]],
        [ 2,  3, 0.50, risk_matrix["L"]],
        [ 2,  6, 0.60, risk_matrix["L"]],
        [ 2,  7, 0.80, risk_matrix["L"]],
        [ 3,  4, 0.55, risk_matrix["L"]], 
        [ 3,  6, 0.80, risk_matrix["L"]],
        [ 3,  7, 0.60, risk_matrix["L"]],
        [ 4,  5, 0.55, risk_matrix["L"]],
        [ 4,  7, 0.80, risk_matrix["L"]],
        [ 4,  8, 0.80, risk_matrix["L"]],
        [ 4, 10, 1.10, risk_matrix["L"]],
        [ 5,  8, 0.60, risk_matrix["L"]],
        [ 6,  7, 0.50, risk_matrix["L"]],
        [ 6,  9, 0.58, risk_matrix["L"]],
        [ 7,  8, 1.20, risk_matrix["L"]],
        [ 7, 10, 0.65, risk_matrix["L"]],
        [ 8, 10, 0.75, risk_matrix["L"]],
        [ 8, 13, 1.05, risk_matrix["L"]],
        [ 9, 11, 0.65, risk_matrix["L"]],
        [ 9, 12, 0.75, risk_matrix["L"]],
        [ 9, 14, 0.85, risk_matrix["L"]],
        [10, 12, 0.60, risk_matrix["L"]],
        [10, 13, 0.55, risk_matrix["L"]],
        [10, 16, 1.20, risk_matrix["L"]],
        [11, 12, 1.40, risk_matrix["L"]],
        [11, 14, 0.40, risk_matrix["L"]],
        [12, 13, 1.15, risk_matrix["L"]],
        [12, 14, 0.65, risk_matrix["L"]],
        [12, 15, 0.70, risk_matrix["L"]],
        [12, 16, 0.65, risk_matrix["L"]],
        [13, 16, 0.85, risk_matrix["L"]],
        [14, 15, 0.30, risk_matrix["L"]],
        [14, 17, 0.45, risk_matrix["L"]],
        [15, 16, 0.70, risk_matrix["L"]],
        [15, 17, 0.40, risk_matrix["L"]],
        [15, 18, 0.90, risk_matrix["L"]],
        [16, 18, 0.50, risk_matrix["L"]],
        [17, 18, 1.30, risk_matrix["L"]],
        [17, 19, 0.90, risk_matrix["L"]],
        [17, 20, 1.60, risk_matrix["L"]],
        [18, 19, 1.60, risk_matrix["L"]],
        [18, 20, 0.70, risk_matrix["L"]],
        [19, 20, 1.20, risk_matrix["L"]]
    ]
    
    return connections
    
def Bungalow(risk_matrix):
    # The connections for the environment compiled as a list of lists. Each list 
    # within a list contains four peices of information: 
    #   1. starting node
    #   2. connecting node
    #   3. Linear distance 
    #   4. Risk
    connections = [
            [1, 2, 0.7,  risk_matrix["L"]],
            [1, 4, 1.2,  risk_matrix["ML"]],
            [1, 8, 2.2,  risk_matrix["L"]],
            [2, 3, 0.8,  risk_matrix["HL"]],
            [2, 4, 1.2,  risk_matrix["L"]],
            [2, 8, 2.8,  risk_matrix["ML"]],
            [4, 5, 0.7,  risk_matrix["HM"]],
            [4, 6, 0.8,  risk_matrix["HM"]],
            [4, 7, 0.7,  risk_matrix["HL"]],
            [4, 8, 1.5,  risk_matrix["ML"]],
            [5, 6, 0.3,  risk_matrix["HL"]],
            [5, 7, 0.4,  risk_matrix["HL"]],
            [6, 7, 0.3,  risk_matrix["HL"]],
            [8, 9, 0.5,  risk_matrix["MH"]],
            [8, 10, 0.8, risk_matrix["HL"]],
            [8, 11, 0.4, risk_matrix["MH"]],
            [8, 12, 1.4, risk_matrix["HM"]],
            [9, 10, 0.7, risk_matrix["HL"]],
            [9, 11, 1.3, risk_matrix["HM"]],
            [9, 12, 1.3, risk_matrix["HM"]],
            [9, 23, 1.1, risk_matrix["HM"]],
            [9, 25, 1.2, risk_matrix["HL"]],
            [9, 26, 1.2, risk_matrix["MH"]],
            [10, 11, 1.0, risk_matrix["HL"]],
            [10, 12, 0.8, risk_matrix["HH"]],
            [10, 23, 0.5, risk_matrix["HH"]],
            [10, 25, 0.8, risk_matrix["HM"]],
            [10, 26, 1.4, risk_matrix["HL"]],
            [11, 12, 0.7, risk_matrix["HM"]],
            [11, 14, 0.7, risk_matrix["M"]],
            [11, 15, 1.2, risk_matrix["MH"]],
            [12, 13, 0.8, risk_matrix["HM"]],
            [12, 19, 0.5, risk_matrix["HM"]],
            [12, 20, 0.4, risk_matrix["VH"]],
            [13, 14, 0.8, risk_matrix["MH"]],
            [13, 18, 0.5, risk_matrix["MH"]],
            [13, 19, 0.5, risk_matrix["HL"]],
            [14, 15, 0.5, risk_matrix["MH"]],
            [14, 16, 0.6, risk_matrix["ML"]],
            [14, 17, 0.6, risk_matrix["ML"]],
            [14, 18, 0.7, risk_matrix["ML"]],
            [15, 16, 0.6, risk_matrix["ML"]],
            [16, 17, 1.0, risk_matrix["M"]],
            [16, 18, 0.8, risk_matrix["ML"]],
            [17, 18, 0.5, risk_matrix["M"]],
            [18, 19, 0.7, risk_matrix["MH"]],
            [19, 21, 1.0, risk_matrix["HM"]],
            [20, 21, 0.7, risk_matrix["VH"]],
            [20, 23, 0.7, risk_matrix["HH"]],
            [21, 22, 0.7, risk_matrix["MH"]],
            [22, 24, 1.0, risk_matrix["MH"]],
            [23, 25, 1.0, risk_matrix["MH"]],
            [24, 25, 1.2, risk_matrix["ML"]],
            [25, 26, 1.4, risk_matrix["ML"]],
            [26, 27, 0.4, risk_matrix["MH"]],
            [26, 28, 0.8, risk_matrix["MH"]],
            [26, 29, 0.6, risk_matrix["ML"]],
            [26, 30, 0.7, risk_matrix["MH"]],
            [27, 28, 0.3, risk_matrix["ML"]],
            [29, 30, 0.4, risk_matrix["ML"]],
        ]
    
    return connections

def CSI_Cobot(risk_matrix):
    # The connections for the environment compiled as a list of lists. Each list 
    # within a list contains four peices of information: 
    #   1. starting node
    #   2. connecting node
    #   3. Linear distance 
    #   4. Risk
    d1 = 0.25
    d2 = 0.50
    d3 = 0.75
    d4 = 1.00
    d5 = 1.50
    d6 = 2.00

    connections = [
        [ 1,  2, d5, risk_matrix["M"]],
        [ 1, 18, d6, risk_matrix["M"]],
        [ 2,  3, d3, risk_matrix["M"]],
        [ 2,  9, d4, risk_matrix["M"]],
        [ 3,  4, d5, risk_matrix["M"]],
        [ 4,  5, d4, risk_matrix["M"]],
        [ 5,  6, d5, risk_matrix["M"]],
        [ 6,  7, d2, risk_matrix["M"]],
        [ 6, 12, d3, risk_matrix["M"]],
        [ 7,  8, d3, risk_matrix["M"]],
        [ 7, 11, d4, risk_matrix["M"]], 
        [ 7, 12, d3, risk_matrix["M"]],
        [ 7, 13, d5, risk_matrix["M"]],
        [ 8,  9, d3, risk_matrix["M"]], 
        [ 8, 10, d4, risk_matrix["M"]],
        [ 8, 11, d3, risk_matrix["M"]], 
        [ 8, 12, d4, risk_matrix["M"]], 
        [ 9, 10, d3, risk_matrix["M"]],
        [10, 11, d3, risk_matrix["M"]],
        [10, 16, d5, risk_matrix["M"]],
        [10, 17, d6, risk_matrix["M"]],
        [10, 18, d6, risk_matrix["M"]],
        [11, 12, d3, risk_matrix["M"]],
        [12, 13, d4, risk_matrix["M"]],
        [13, 14, d5, risk_matrix["M"]],
        [14, 15, d3, risk_matrix["M"]], 
        [14, 25, d4, risk_matrix["M"]], 
        [14, 26, d4, risk_matrix["M"]], 
        [14, 27, d4, risk_matrix["M"]],
        [14, 28, d5, risk_matrix["M"]],
        [15, 16, d3, risk_matrix["M"]], 
        [15, 23, d3, risk_matrix["M"]], 
        [15, 25, d3, risk_matrix["M"]], 
        [15, 26, d4, risk_matrix["M"]], 
        [16, 17, d3, risk_matrix["M"]], 
        [16, 22, d4, risk_matrix["M"]], 
        [16, 23, d3, risk_matrix["M"]], 
        [17, 18, d3, risk_matrix["M"]], 
        [17, 19, d1, risk_matrix["M"]],
        [18, 19, d2, risk_matrix["M"]],
        [19, 20, d5, risk_matrix["M"]], 
        [20, 21, d3, risk_matrix["M"]], 
        [20, 33, d4, risk_matrix["M"]],
        [21, 22, d3, risk_matrix["M"]],
        [21, 23, d2, risk_matrix["M"]],
        [21, 24, d3, risk_matrix["M"]],
        [21, 31, d2, risk_matrix["M"]],
        [22, 23, d1, risk_matrix["M"]], 
        [22, 24, d3, risk_matrix["M"]], 
        [23, 24, d2, risk_matrix["M"]], 
        [23, 25, d3, risk_matrix["M"]], 
        [24, 25, d3, risk_matrix["M"]], 
        [24, 30, d4, risk_matrix["M"]],
        [24, 31, d2, risk_matrix["M"]],
        [25, 26, d3, risk_matrix["M"]], 
        [25, 30, d2, risk_matrix["M"]],
        [26, 27, d4, risk_matrix["M"]],
        [26, 28, d4, risk_matrix["M"]],
        [26, 29, d3, risk_matrix["M"]],
        [27, 28, d3, risk_matrix["M"]],
        [28, 29, d4, risk_matrix["M"]], 
        [28, 38, d5, risk_matrix["M"]],
        [29, 30, d4, risk_matrix["M"]], 
        [29, 37, d5, risk_matrix["M"]], 
        [29, 38, d4, risk_matrix["M"]],
        [30, 31, d5, risk_matrix["M"]],
        [30, 32, d5, risk_matrix["M"]],
        [31, 32, d2, risk_matrix["M"]],
        [32, 33, d3, risk_matrix["M"]],
        [32, 34, d4, risk_matrix["M"]],
        [32, 35, d3, risk_matrix["M"]],
        [33, 34, d3, risk_matrix["M"]],
        [33, 35, d4, risk_matrix["M"]], 
        [34, 35, d4, risk_matrix["M"]],
        [35, 36, d5, risk_matrix["M"]],
        [36, 37, d5, risk_matrix["M"]],
        [37, 38, d3, risk_matrix["M"]],

    ]


    return connections
