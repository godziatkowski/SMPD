package classifier;

import java.util.*;
import java.util.stream.Collectors;

public class KNNClassifier implements IKClassifier {

    private Map<String, double[][]> trainingSets;
    private Map<String, double[][]> testSets;
    private Set<String> keysInTrainingSets;
    private Set<String> keysInTestSets;

    private void getNamesOfTrainingAndTestSets() {
        keysInTestSets = testSets.keySet();
        keysInTrainingSets = trainingSets.keySet();
    }

    private void splitPassedSetsIntoTestAndTraining( Map<String, double[][]> probesSplitedIntoTrainingAndTestSets ) {
        trainingSets = new HashMap<>();
        testSets = new HashMap<>();

        probesSplitedIntoTrainingAndTestSets.keySet().forEach( className -> {
            if ( className.contains( "_test" ) ) {
                testSets.put( className, probesSplitedIntoTrainingAndTestSets.get( className ) );
            } else {
                trainingSets.put( className, probesSplitedIntoTrainingAndTestSets.get( className ) );
            }
        } );
        getNamesOfTrainingAndTestSets();
    }

    @Override
    public double train( Map<String, double[][]> probesSplitedIntoTrainingAndTestSets, Set<Integer> indexesOfBestAttributes, int kCount ) {
        int countOfFailedAssignments = 0;
        int countOfSuccessAssignments = 0;
        splitPassedSetsIntoTestAndTraining( probesSplitedIntoTrainingAndTestSets );

        for ( String keyInTestSet : keysInTestSets ) {
            for ( int testProbeIndex = 0; testProbeIndex < testSets.get( keyInTestSet )[0].length; testProbeIndex++ ) {
                int probeIndex = testProbeIndex;
                List<ProbeDistance> distancesBetweenProbeAndNearestClassNeighbour = new ArrayList<>();
                keysInTrainingSets.stream().forEach( ( keyInTrainingSet ) -> {
                    List<Double> closestClassNeighbours = new ArrayList<>();
                    for ( int trainingProbeIndex = 0; trainingProbeIndex < trainingSets.get( keyInTrainingSet )[0].length; trainingProbeIndex++ ) {
                        double distanceToProbe = 0.0;
                        for ( int attributeIndex = 0; attributeIndex < trainingSets.get( keyInTrainingSet ).length; attributeIndex++ ) {
                            if ( indexesOfBestAttributes.isEmpty() || indexesOfBestAttributes.contains( attributeIndex ) ) {
                                distanceToProbe += Math.pow( testSets.get( keyInTestSet )[attributeIndex][probeIndex] - trainingSets.get( keyInTrainingSet )[attributeIndex][trainingProbeIndex], 2 );
                            }
                        }
                        distanceToProbe = Math.sqrt( distanceToProbe );
                        if ( closestClassNeighbours.size() < kCount ) {
                            closestClassNeighbours.add( distanceToProbe );
                        } else {
                            Collections.sort( closestClassNeighbours );

                            if ( distanceToProbe < closestClassNeighbours.get( closestClassNeighbours.size() - 1 ) ) {
                                closestClassNeighbours.set( closestClassNeighbours.size() - 1, distanceToProbe );
                            }
                        }
                    }
                    List<ProbeDistance> smallestProbeDistances = closestClassNeighbours.stream()
                            .map( distance -> {
                                return new ProbeDistance( keyInTrainingSet, distance );
                            } ).collect( Collectors.toList() );

                    distancesBetweenProbeAndNearestClassNeighbour.addAll( smallestProbeDistances );
                } );

                String closestClass = "";
                Integer closestClassCount = null;
                Map<String, Integer> countOfClosestNegihboursPerClass = new HashMap<>();

                Collections.sort( distancesBetweenProbeAndNearestClassNeighbour );

                for ( int i = 0; i < kCount; i++ ) {
                    ProbeDistance probeDistance = distancesBetweenProbeAndNearestClassNeighbour.get( i );
                    if ( countOfClosestNegihboursPerClass.containsKey( probeDistance.getClassName() ) ) {
                        countOfClosestNegihboursPerClass.put( probeDistance.getClassName(), ( countOfClosestNegihboursPerClass.get( probeDistance.getClassName() ) + 1 ) );
                    } else {
                        countOfClosestNegihboursPerClass.put( probeDistance.getClassName(), 1 );
                    }
                }

                for ( String keyInCountOfClosestNegihboursPerClass : countOfClosestNegihboursPerClass.keySet() ) {
                    if ( closestClassCount == null || closestClassCount < countOfClosestNegihboursPerClass.get( keyInCountOfClosestNegihboursPerClass ) ) {
                        closestClass = keyInCountOfClosestNegihboursPerClass;
                        closestClassCount = countOfClosestNegihboursPerClass.get( keyInCountOfClosestNegihboursPerClass );
                    }
                }

                if ( keyInTestSet.contains( closestClass.replace( "_training", "" ).trim() ) ) {
                    countOfSuccessAssignments++;
                } else {
                    countOfFailedAssignments++;
                }
            }
        }
        double percentageAlgorithmCorrectness = ( (double) countOfSuccessAssignments / (double) ( countOfFailedAssignments + countOfSuccessAssignments ) ) * 100;
        return percentageAlgorithmCorrectness;
    }

}
