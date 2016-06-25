package classifier;

import java.util.Map;
import java.util.Set;

public class NNClassifier implements IClassifier {

    @Override
    public double train( Map<String, double[][]> probesSplitedIntoTrainingAndTestSets, Set<Integer> indexesOfBestAttributes ) {
        IKClassifier iKClassifier = new KNNClassifier();
        return iKClassifier.train( probesSplitedIntoTrainingAndTestSets, indexesOfBestAttributes, 1 );
    }

}
