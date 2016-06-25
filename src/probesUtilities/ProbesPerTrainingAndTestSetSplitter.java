package probesUtilities;

import Jama.Matrix;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ProbesPerTrainingAndTestSetSplitter {

    private static final int MAXIMUM_BOOTSTRAP_TRIALS = 20;
    private static final String _TRAINING = "_training";
    private static final String _TEST = "_test";

    public static List<Map<String, double[][]>> simpleValidationProbeSplitter( Map<String, double[][]> probesGroupedByClass, double percentageDistribution ) {
        Map<String, double[][]> mapWithSplittedProbes = new HashMap<>();
        Random random = new Random( LocalDateTime.now().getNano() );

        probesGroupedByClass.entrySet().forEach( entry -> {
            String testSetName = entry.getKey() + _TEST;
            String trainingSetName = entry.getKey() + _TRAINING;
            int countOfProbesForTestSet = (int) Math.round( entry.getValue()[0].length * ( 1 - percentageDistribution ) );
            int countOfAttributes = entry.getValue().length;
            Set<Integer> indexesOfTestProbes = new HashSet<>();
            while ( indexesOfTestProbes.size() < countOfProbesForTestSet ) {
                indexesOfTestProbes.add( random.nextInt( entry.getValue()[0].length ) );
            }

            double[][] testSet = new double[countOfAttributes][countOfProbesForTestSet];
            double[][] trainingSet = new double[countOfAttributes][entry.getValue()[0].length - countOfProbesForTestSet];
            int testSetIndex = 0;
            int trainingSetIndex = 0;

            for ( int probeIndex = 0; probeIndex < entry.getValue()[0].length; probeIndex++ ) {
                if ( indexesOfTestProbes.contains( probeIndex ) ) {
                    for ( int attributeIndex = 0; attributeIndex < countOfAttributes; attributeIndex++ ) {
                        testSet[attributeIndex][testSetIndex] = entry.getValue()[attributeIndex][probeIndex];
                    }
                    testSetIndex++;
                } else {
                    for ( int attributeIndex = 0; attributeIndex < countOfAttributes; attributeIndex++ ) {
                        trainingSet[attributeIndex][trainingSetIndex] = entry.getValue()[attributeIndex][probeIndex];
                    }
                    trainingSetIndex++;
                }
            }
            mapWithSplittedProbes.put( testSetName, testSet );
            mapWithSplittedProbes.put( trainingSetName, trainingSet );
        } );
        List<Map<String, double[][]>> result = new ArrayList<>();
        result.add( mapWithSplittedProbes );
        return result;
    }

    public static List<Map<String, double[][]>> crossValidationProbeSplitter( double[][] probes, int setsCount, String[] classNames, int[] countOfSamplesPerClass ) {
        int countOfProbesPerEachSection = probes[0].length / setsCount;
        int overallProbeCount = probes[0].length;
        List<Map<String, double[][]>> setsOfProbes = new ArrayList<>();

        List<Integer> probeIndexes = IntStream.rangeClosed( 0, overallProbeCount - 1 ).boxed().collect( Collectors.toList() );
        Collections.shuffle( probeIndexes );

        for ( int setIndex = 0; setIndex < setsCount; setIndex++ ) {
            Map<String, double[][]> splitedProbes = new HashMap<>();
            Map<String, Integer> addedProbesToEachSet = new HashMap<>();

            List<Integer> testProbesIndexes = probeIndexes.stream()
                    .skip( setIndex * countOfProbesPerEachSection )
                    .limit( countOfProbesPerEachSection )
                    .collect( Collectors.toList() );
            int[] countOfTestProbesInEachClass = getCountOfProbesInSetForClasses( testProbesIndexes, countOfSamplesPerClass );

            for ( int index = 0; index < classNames.length; index++ ) {
                String className = classNames[index];
                splitedProbes.put( className + _TRAINING, new double[overallProbeCount - countOfTestProbesInEachClass[index]][probes.length] );
                splitedProbes.put( className + _TEST, new double[countOfTestProbesInEachClass[index]][probes.length] );
                addedProbesToEachSet.put( className + _TRAINING, 0 );
                addedProbesToEachSet.put( className + _TEST, 0 );
            }
            probeIndexes.stream().forEach( ( probeIndex ) -> {
                String className;
                int classIndex = getClassIndex( probeIndex, countOfSamplesPerClass );
                if ( testProbesIndexes.contains( probeIndex ) ) {
                    className = classNames[classIndex] + _TEST;
                } else {
                    className = classNames[classIndex] + _TRAINING;
                }
                double[] probe = MatrixUtilities.getProbe( probes, probeIndex );
                int probeIndexInSplittedArray = addedProbesToEachSet.get( className );
                splitedProbes.get( className )[probeIndexInSplittedArray] = probe;
                probeIndexInSplittedArray++;
                addedProbesToEachSet.put( className, probeIndexInSplittedArray );
            } );

            splitedProbes.keySet().stream().forEach( ( key ) -> {
                double[][] transponsedMatrix = new Matrix( splitedProbes.get( key ) ).transpose().getArray();
                splitedProbes.put( key, transponsedMatrix );
            } );
            setsOfProbes.add( splitedProbes );
        }

        return setsOfProbes;
    }

    public static List<Map<String, double[][]>> bootstrapValidationProbeSplitter( double[][] probes, int setsCount, String[] classNames, int[] countOfSamplesPerClass ) {
        int overallProbeCount = probes[0].length;
        List<Map<String, double[][]>> setsOfProbes = new ArrayList<>();
        Random random = new Random( LocalDateTime.now().getNano() );
        for ( int bootstrapTrialIndex = 0; bootstrapTrialIndex < MAXIMUM_BOOTSTRAP_TRIALS; bootstrapTrialIndex++ ) {
            Map<String, double[][]> splitedProbes = new HashMap<>();
            Map<String, Integer> addedProbesToEachSet = new HashMap<>();

            List<Integer> trainingProbeIndexes = new ArrayList<>( overallProbeCount );
            List<Integer> testProbeIndexes = new ArrayList<>();

            for ( int probeIndex = 0; probeIndex < overallProbeCount; probeIndex++ ) {
                int randomProbeIndex = random.nextInt( overallProbeCount );
                trainingProbeIndexes.add( randomProbeIndex );
            }
            for ( int testProbeIndex = 0; testProbeIndex < overallProbeCount; testProbeIndex++ ) {
                if ( !trainingProbeIndexes.contains( testProbeIndex ) ) {
                    testProbeIndexes.add( testProbeIndex );
                }
            }

            int[] countOfTrainingProbesInEachClass = getCountOfProbesInSetForClasses( trainingProbeIndexes, countOfSamplesPerClass );
            int[] countOfTestProbesInEachClass = getCountOfProbesInSetForClasses( testProbeIndexes, countOfSamplesPerClass );
            for ( int index = 0; index < classNames.length; index++ ) {
                String className = classNames[index];
                splitedProbes.put( className + _TRAINING, new double[countOfTrainingProbesInEachClass[index]][probes.length] );
                splitedProbes.put( className + _TEST, new double[countOfTestProbesInEachClass[index]][probes.length] );
                addedProbesToEachSet.put( className + _TRAINING, 0 );
                addedProbesToEachSet.put( className + _TEST, 0 );
            }
            trainingProbeIndexes.stream().forEach( ( probeIndex ) -> {
                double[] probe = MatrixUtilities.getProbe( probes, probeIndex );
                int classIndex = getClassIndex( probeIndex, countOfSamplesPerClass );
                String className = classNames[classIndex] + _TRAINING;
                int probeIndexInSplittedArray = addedProbesToEachSet.get( className );
                splitedProbes.get( className )[probeIndexInSplittedArray] = probe;
                probeIndexInSplittedArray++;
                addedProbesToEachSet.put( className, probeIndexInSplittedArray );
            } );
            testProbeIndexes.stream().forEach( ( probeIndex ) -> {
                double[] probe = MatrixUtilities.getProbe( probes, probeIndex );
                int classIndex = getClassIndex( probeIndex, countOfSamplesPerClass );
                String className = classNames[classIndex] + _TEST;
                int probeIndexInSplittedArray = addedProbesToEachSet.get( className );
                splitedProbes.get( className )[probeIndexInSplittedArray] = probe;
                probeIndexInSplittedArray++;
                addedProbesToEachSet.put( className, probeIndexInSplittedArray );
            } );
            
            splitedProbes.keySet().stream().forEach( ( key ) -> {
                double[][] transponsedMatrix = new Matrix( splitedProbes.get( key ) ).transpose().getArray();
                splitedProbes.put( key, transponsedMatrix );
            } );
            setsOfProbes.add( splitedProbes );

        }
        return setsOfProbes;
    }

    private static int getClassIndex( Integer probeIndex, int[] countOfSamplesPerClass ) {
        if ( probeIndex < countOfSamplesPerClass[0] ) {
            return 0;
        } else {
            return 1;
        }
    }

    private static void printInt( List<Integer> ints ) {
        StringBuilder sb = new StringBuilder( "indexes: " );
        for ( Integer aInt : ints ) {
            sb.append( aInt ).append( " " );
        }
        System.out.println( sb.toString() );
    }

    private static void printInt( int[] ints ) {
        StringBuilder sb = new StringBuilder( "indexes: " );
        for ( Integer aInt : ints ) {
            sb.append( aInt ).append( " " );
        }
        System.out.println( sb.toString() );
    }

    private static void printProbe( double[] probe ) {
        StringBuilder sb = new StringBuilder( "indexes: " );
        for ( Double aInt : probe ) {
            sb.append( aInt ).append( " " );
        }
        System.out.println( sb.toString() );
    }

    private static int[] getCountOfProbesInSetForClasses( List<Integer> probesIndexes, int[] countOfSamplesPerClass ) {
        int[] countOfTestProbesInEachClass = new int[countOfSamplesPerClass.length];
        int classIndex;
        for ( Integer probesIndex : probesIndexes ) {
            classIndex = getClassIndex( probesIndex, countOfSamplesPerClass );
            countOfTestProbesInEachClass[classIndex]++;
        }

        return countOfTestProbesInEachClass;
    }
}
