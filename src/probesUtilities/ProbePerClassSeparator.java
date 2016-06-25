package probesUtilities;

import java.util.HashMap;
import java.util.Map;

public class ProbePerClassSeparator {
    
    public Map<String, double[][]> probesGroupedByClass;

    public Map<String, double[][]> separateProbesPerClass( String[] classNames, int featureCount, int[] countOfSamplesPerClass, double[][] mixedProbes ) {
        probesGroupedByClass = new HashMap<>();
        int indexOfProbe = 0;
        for ( int i = 0; i < classNames.length; i++ ) {
            String className = classNames[i];
            double[][] temporaryArray = new double[featureCount][countOfSamplesPerClass[i]];
            int probeIndexInClassMatrix = 0;
            for ( ; indexOfProbe < countOfSamplesPerClass[i]; indexOfProbe++ ) {
                for( int attribute =0; attribute < featureCount; attribute++){
                    temporaryArray[attribute][probeIndexInClassMatrix] = mixedProbes[attribute][indexOfProbe];
                }
                probeIndexInClassMatrix++;
            }
            probesGroupedByClass.put( className, temporaryArray );
        }
        return probesGroupedByClass;
    }

}
