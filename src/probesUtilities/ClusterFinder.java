package probesUtilities;

import Jama.Matrix;
import exceptions.EmptyClusterException;
import java.util.*;

import static probesUtilities.EuklideanDistance.calculateEuklideanDistanceBetweenMatrixes;

public class ClusterFinder {

    private static final double MAXIMAL_ACCEPTABLE_CLUSTER_CENTROID_DISLOCATION = 0.000001;
    private static final double H = 0.1;

    public Map<Matrix, Matrix> assignProbesToClusters( Matrix allProbesInClass ) {
        Map<Matrix, Matrix> probesGroupedToClusters = new HashMap<>();
        List<Matrix> clustersCentroids = findOptimalClusters( allProbesInClass );
        Map<Matrix, List<double[]>> probesAssignedToClusters = assignSamplesToMods( allProbesInClass, clustersCentroids );
        probesAssignedToClusters.entrySet().forEach( entry -> {
            probesGroupedToClusters.put( entry.getKey(), convertListOfProbesToMatrix( entry.getValue() ) );
        } );
        return probesGroupedToClusters;
    }

    private List<Matrix> findOptimalClusters( Matrix allProbesInClass ) {
        List<Double> errors = new ArrayList<>();        
        Map<Integer, List<Matrix>> modCentersByNumberOfMods = new HashMap<>();
        int modsCount = 0;        
        do {
            try {
                modsCount += 1;
                
                List<Matrix> modCenters = findCentroidsForNClusters( allProbesInClass, modsCount );
                modCentersByNumberOfMods.put( modsCount, modCenters );
                errors.add( calculateError( modCenters, allProbesInClass ) );
            } catch ( EmptyClusterException e ) {
                System.out.println( "Exception" );
                break;
            }
        } while ( modsCount < 2 || errorDecreasedSignificantly( errors, modsCount ) );
        return modCentersByNumberOfMods.get( modsCount - 1 );
    }

    private Map<Matrix, List<double[]>> assignSamplesToMods( Matrix allProbesInClass, List<Matrix> clustersCentroids ) {
        Map<Matrix, List<double[]>> probesAssignedToMods = new HashMap<>();
        clustersCentroids.stream().forEach( ( clustersCentroid ) -> {
            probesAssignedToMods.put( clustersCentroid, new ArrayList<>() );
        } );
        for ( int i = 0; i < allProbesInClass.getColumnDimension(); i++ ) {
            double[] probe = MatrixUtilities.getProbe( allProbesInClass.getArray(), i );
            Matrix probeMatrix = new Matrix( probe, 1 );
            Matrix nearestClusterCentroid = null;
            double minimalDistance = 0.0;
            for ( Matrix clusterCentroid : clustersCentroids ) {
                double distance = calculateEuklideanDistanceBetweenMatrixes( probeMatrix, clusterCentroid );
                if ( nearestClusterCentroid == null || minimalDistance > distance ) {
                    minimalDistance = distance;
                    nearestClusterCentroid = clusterCentroid;
                }
            }
            probesAssignedToMods.get( nearestClusterCentroid ).add( probe );
        }
        return probesAssignedToMods;
    }

    private boolean errorDecreasedSignificantly( List<Double> errors, int numberOfMods ) {
        return Math.abs( errors.get( numberOfMods - 2 ) - errors.get( numberOfMods - 1 ) ) > errors.get( numberOfMods - 2 ) * H;
    }

    private List<Matrix> findCentroidsForNClusters( Matrix allProbesInClass, int clusterCount ) {
        List<Matrix> optimalClusterCentroids = new ArrayList<>();
        if ( clusterCount > 1 ) {
            List<Matrix> clustersCentroids = randomModCenters( allProbesInClass, clusterCount );

            double maxCentroidDislocation;
            do {
                Map<Matrix, List<double[]>> samplesGroupedByMods = assignSamplesToMods( allProbesInClass, clustersCentroids );

                List<Matrix> newClustersCentroids = correctModCenters( clustersCentroids, samplesGroupedByMods );
                maxCentroidDislocation = calculateMaximumDislocationOfClustersCentroids( clustersCentroids, newClustersCentroids );

                clustersCentroids = newClustersCentroids;
            } while ( maxCentroidDislocation > MAXIMAL_ACCEPTABLE_CLUSTER_CENTROID_DISLOCATION );
            optimalClusterCentroids = clustersCentroids;
        } else {
            optimalClusterCentroids.add( calculateMeanValueForMatrix( allProbesInClass ) );
        }
        return optimalClusterCentroids;
    }

    private Matrix calculateMeanValueForMatrix( Matrix matrix ) {
        double[][] probes = matrix.getArray();
        double[] meanAttributes = new double[matrix.getRowDimension()];

        for ( int attributeIndex = 0; attributeIndex < meanAttributes.length; attributeIndex++ ) {
            double mean = 0.0;
            for ( int probeIndex = 0; probeIndex < probes[attributeIndex].length; probeIndex++ ) {
                mean += probes[attributeIndex][probeIndex];
            }
            meanAttributes[attributeIndex] = mean / probes[attributeIndex].length;

        }
        return new Matrix( meanAttributes, 1 );
    }

    private List<Matrix> randomModCenters( Matrix probes, int numberOfClusters ) {
        Set<Integer> choosenProbesIndexes = new HashSet<>();
        List<Matrix> clusters = new ArrayList<>();
        for ( int i = 0; i < numberOfClusters; i++ ) {
            int probeIndex;
            do {
                probeIndex = new Random().nextInt( probes.getColumnDimension() );
            } while ( choosenProbesIndexes.contains( probeIndex ) );
            choosenProbesIndexes.add( probeIndex );
            double[] probe = MatrixUtilities.getProbe( probes.getArray(), probeIndex );

            clusters.add( new Matrix( probe, 1 ) );
        }
        return clusters;
    }

    private List<Matrix> correctModCenters( List<Matrix> clustersCentroids, Map<Matrix, List<double[]>> samplesGroupedByMods ) {
        List<Matrix> newClusterCentroids = new ArrayList<>();
        for ( Matrix clusterCentroid : clustersCentroids ) {
            List<double[]> probesInCluster = samplesGroupedByMods.get( clusterCentroid );
            if ( probesInCluster.isEmpty() ) {
                throw new EmptyClusterException();
            }
            Matrix probes = convertListOfProbesToMatrix( probesInCluster );
            Matrix newClusterCentroid = calculateMeanValueForMatrix( probes );
            newClusterCentroids.add( newClusterCentroid );
        }
        return newClusterCentroids;
    }

    private Matrix convertListOfProbesToMatrix( List<double[]> probesAsList ) {
        double[][] probes = new double[probesAsList.get( 0 ).length][probesAsList.size()];
        for ( int probeIndex = 0; probeIndex < probesAsList.size(); probeIndex++ ) {
            for ( int attributeIndex = 0; attributeIndex < probes.length; attributeIndex++ ) {
                probes[attributeIndex][probeIndex] = probesAsList.get( probeIndex )[attributeIndex];
            }
        }
        return new Matrix( probes );
    }

    private double calculateMaximumDislocationOfClustersCentroids( List<Matrix> clustersCentroids, List<Matrix> newClustersCentroids ) {
        double maximumDislocation = 0;
        for ( int clusterIndex = 0; clusterIndex < clustersCentroids.size(); clusterIndex++ ) {
            double distanceBetweenOldAndNewClusterCentroid
                   = calculateEuklideanDistanceBetweenMatrixes( clustersCentroids.get( clusterIndex ), newClustersCentroids.get( clusterIndex ) );
            maximumDislocation = Math.max( maximumDislocation, distanceBetweenOldAndNewClusterCentroid );
        }
        return maximumDislocation;
    }

    private Double calculateError( List<Matrix> clusterCentroids, Matrix allProbesInClass ) {        
        Map<Matrix, List<double[]>> probesGroupedToClusters = assignSamplesToMods( allProbesInClass, clusterCentroids );
        double error = 0;

        for ( Matrix clusterCentroid : clusterCentroids ) {
            List<double[]> probesInCluster = probesGroupedToClusters.get( clusterCentroid );
            double sum = probesInCluster
                    .stream()
                    .mapToDouble( probe -> calculateEuklideanDistanceBetweenMatrixes( clusterCentroid, new Matrix( probe, 1 ) ) )
                    .sum();
            error += sum / probesInCluster.size();
        }

        return error / clusterCentroids.size();
    }

    private void printMatrix( Matrix matrix ) {
        System.out.println( "rows: " + matrix.getRowDimension() + " cols: " + matrix.getColumnDimension() );
        double[][] matrixAsArray = matrix.getArray();
        for ( double[] row : matrixAsArray ) {
            printArrayOneDim( row );
        }
    }

    private void printArrayOneDim( double[] array ) {
        String row = "";
        for ( double element : array ) {
            row += element + " ";
        }
        System.out.println( row );
    }

}
