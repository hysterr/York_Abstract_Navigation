def Risk():
    risk_matrix = {
        "L"  : 0.95,
        "ML" : 0.90,
        "M"  : 0.87,
        "MH" : 0.85,
        "HL" : 0.80,
        "HM" : 0.75,
        "H"  : 0.65,
        "HH" : 0.50,
        "VH" : 0.40, 
    }
    
    return risk_matrix


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
            [11, 12, 1.0, risk_matrix["HM"]],
            [11, 15, 1.7, risk_matrix["MH"]],
            [12, 13, 0.6, risk_matrix["HM"]],
            [12, 19, 0.5, risk_matrix["HM"]],
            [12, 20, 0.4, risk_matrix["VH"]],
            [13, 18, 0.2, risk_matrix["MH"]],
            [13, 19, 0.5, risk_matrix["HL"]],
            [14, 15, 0.5, risk_matrix["MH"]],
            [14, 16, 0.6, risk_matrix["ML"]],
            [14, 17, 0.7, risk_matrix["ML"]],
            [14, 18, 0.9, risk_matrix["ML"]],
            [15, 16, 0.7, risk_matrix["ML"]],
            [16, 17, 1.0, risk_matrix["M"]],
            [16, 18, 0.8, risk_matrix["ML"]],
            [17, 18, 0.5, risk_matrix["M"]],
            [18, 19, 0.4, risk_matrix["MH"]],
            [19, 21, 1.0, risk_matrix["HM"]],
            [20, 21, 0.7, risk_matrix["VH"]],
            [20, 23, 0.7, risk_matrix["HH"]],
            [21, 22, 0.7, risk_matrix["HM"]],
            [22, 24, 1.0, risk_matrix["HL"]],
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