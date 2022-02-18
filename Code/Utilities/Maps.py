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
            [10, 11, 1.2, risk_matrix["HL"]],
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
            [13, 14, 0.6, risk_matrix["MH"]],
            [13, 18, 0.2, risk_matrix["MH"]],
            [13, 19, 0.5, risk_matrix["HL"]],
            [14, 15, 0.5, risk_matrix["MH"]],
            [14, 16, 0.6, risk_matrix["ML"]],
            [14, 17, 0.6, risk_matrix["ML"]],
            [14, 18, 0.7, risk_matrix["ML"]],
            [15, 16, 0.6, risk_matrix["ML"]],
            [16, 17, 1.0, risk_matrix["M"]],
            [16, 18, 0.8, risk_matrix["ML"]],
            [17, 18, 0.5, risk_matrix["M"]],
            [18, 19, 0.4, risk_matrix["MH"]],
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
