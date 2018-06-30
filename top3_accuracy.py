import numpy as np

def top3_tocnost(predictions, test_generator):
    top3_ind = []
    for row in predictions:
        idx = []
        temp = row
        for i in range(3):
            max_idx = np.argmax(temp, axis=0)
            idx.append(max_idx)
            temp = np.delete(temp, max_idx)
        top3_ind.append(idx)
        
    suma = 0
    for i in range(len(top3_ind)):
        if test_generator.classes[i] in top3_ind[i]:
            suma += 1

    tocnost = suma/len(top3_ind)
    print("tocnost top-3 klasifikacije " + str(tocnost*100) + " %")
    
    return tocnost
