package classifier;

import java.util.Map;
import java.util.Set;

//Interfejs dla klasyfikatora KNN
public interface IKClassifier{

    /**
     * Klasyfikator KNN potrzebuje dodatkowego parametru dlatego też otrzymał własny interfejs
     * @param probesSplitedIntoTrainingAndTestSets (patrz interfejs IClassifier)
     * @param indexesOfBestAttributes (patrz interfejs IClassifier)
     * @param kCount - parametr określający jak duża powinna być grupa najbliższych sąsiadów
     * @return (patrz interfejs IClassifier)
     */
    double train( Map<String, double[][]> probesSplitedIntoTrainingAndTestSets, Set<Integer> indexesOfBestAttributes, int kCount );
    
}
