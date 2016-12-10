package classifier;

import java.util.Map;
import java.util.Set;

public class NNClassifier implements IClassifier {
    
    /**Parametry metody opisane w interfejsie
     * 
     * @param probesSplitedIntoTrainingAndTestSets
     * @param indexesOfBestAttributes
     * @return 
     */
    @Override
    public double train( Map<String, double[][]> probesSplitedIntoTrainingAndTestSets, Set<Integer> indexesOfBestAttributes ) {
        //Jako że klasyfikacja NN polega na tym samym co klasyfikacja KNN z ograniczeniem ilości najbliższych sąsiadów do jednego to:
        //tworzymy nowy klasyfikator KNN
        IKClassifier iKClassifier = new KNNClassifier();
        //i odelegowujemy do niego zadanie przetrenowania próbek
        return iKClassifier.train( probesSplitedIntoTrainingAndTestSets, indexesOfBestAttributes, 1 );
    }

}
