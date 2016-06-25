package classifier;

import java.util.Map;
import java.util.Set;

public interface IKClassifier{

    double train( Map<String, double[][]> probesSplitedIntoTrainingAndTestSets, Set<Integer> indexesOfBestAttributes, int kCount );
    
}
