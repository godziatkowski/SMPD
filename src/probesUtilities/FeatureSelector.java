package probesUtilities;

import Jama.Matrix;
import exceptions.TooManyDimensionException;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import org.apache.commons.math3.util.Combinations;

import static java.util.stream.Collectors.toSet;
import static probesUtilities.EuklideanDistance.calculateEuklideanDistanceBetweenMatrixes;

public class FeatureSelector {

    private Map<String, Matrix> covarianceMatrixesPerClass;
    private Map<String, double[]> meanValuesPerClass;
    private List<String> classNames;

    public Set<Integer> getBestAttribtesUsingFisher( Map<String, double[][]> probesGroupedByClass, int numberOfAttributes ) {
        classNames = probesGroupedByClass.keySet().stream().collect( Collectors.toList() );
        calculateMeanValuesForEachClass( probesGroupedByClass );
        calculateCovarianceMatrixForEachClass( probesGroupedByClass, meanValuesPerClass );
        Set<Integer> bestAttributeIndexes = findBestAttributes( numberOfAttributes, probesGroupedByClass );
        return bestAttributeIndexes;
    }

    public Set<Integer> selectBestFeatureIndexesUsingSFS( Map<String, double[][]> probesGroupedByClass, int numberOfFeaturesToSelect, int[] ClassLabels, int [] SampleCount ) {
        List<Integer> bestFeatureIndexes = new ArrayList<>( numberOfFeaturesToSelect );
        classNames = probesGroupedByClass.keySet().stream().collect( Collectors.toList() );
        calculateMeanValuesForEachClass( probesGroupedByClass );
        calculateCovarianceMatrixForEachClass( probesGroupedByClass, meanValuesPerClass );
        int featureCount = probesGroupedByClass.get( probesGroupedByClass.keySet().iterator().next()).length;
        double FLD = 0, tmp;
        int bestFeatureIndex = -1;
        double[][] combinedProbes = combineProbesToOneMatrix( probesGroupedByClass );
        for ( int i = 0; i < featureCount; i++ ) {
            if ( ( tmp = computeFisherLD( combinedProbes[i], ClassLabels, SampleCount ) ) > FLD ) {
                FLD = tmp;
                bestFeatureIndex = i;
            }
        }
        bestFeatureIndexes.add( bestFeatureIndex );

        for ( int i = 1; i < numberOfFeaturesToSelect; i++ ) {
            double fisherDiscriminant = Double.MIN_VALUE;
            bestFeatureIndexes.add( -1 );

            for ( int j = 0; j < combinedProbes.length; j++ ) {
                if ( bestFeatureIndexes.contains( j ) ) {
                    continue;
                }

                int[] featureIndexes = new int[i + 1];
                for ( int k = 0; k < i; k++ ) {
                    featureIndexes[k] = bestFeatureIndexes.get( k );
                }
                featureIndexes[i] = j;

                tmp = computeFisherLD( featureIndexes, probesGroupedByClass );
                if ( tmp > fisherDiscriminant ) {
                    fisherDiscriminant = tmp;
                    bestFeatureIndexes.set( i, j );
                }
            }
        }

        return bestFeatureIndexes.stream().collect( Collectors.toSet() );
    }

    private Set<Integer> findBestAttributes( int numberOfAttributes, Map<String, double[][]> probesGroupedByClass )
            throws TooManyDimensionException {
        double[][] combinedProbes = combineProbesToOneMatrix( probesGroupedByClass );
        int[] bestFeatureIndexes = null;
        double fisherDiscriminant = Double.MIN_VALUE;

        Combinations combinations = new Combinations( combinedProbes.length, numberOfAttributes );
        for ( int[] combination : combinations ) {
            double tmp = computeFisherLD( combination, probesGroupedByClass );
            if ( tmp > fisherDiscriminant ) {
                fisherDiscriminant = tmp;
                bestFeatureIndexes = combination;
            }
        }

        return IntStream.of( bestFeatureIndexes )
                .boxed()
                .collect( toSet() );
    }

    private double computeFisherLD( int[] featureIndexes, Map<String, double[][]> probesGroupedByClass ) {

        Map<String, Matrix> probesWithExtractedFeatures = new HashMap<>();
        probesGroupedByClass.entrySet().stream().forEach( ( entrySet ) -> {
            Matrix probesWithExtractRequiredFeatures = extractRequiredFeatures( new Matrix( entrySet.getValue() ), featureIndexes );
            probesWithExtractedFeatures.put( entrySet.getKey(), probesWithExtractRequiredFeatures );
        } );

        Matrix meanOfFirstClass = extractRequiredFeatures( meanValuesPerClass.get( classNames.get( 0 ) ), featureIndexes );
        Matrix meanOfSecondClass = extractRequiredFeatures( meanValuesPerClass.get( classNames.get( 1 ) ), featureIndexes );

        Matrix covarianceMatrixOfFirstClass = calculateCovarianceMatrix( probesWithExtractedFeatures.get( classNames.get( 0 ) ), meanOfFirstClass );
        Matrix covarianceMatrixOfSecondClass = calculateCovarianceMatrix( probesWithExtractedFeatures.get( classNames.get( 1 ) ), meanOfSecondClass );

        return calculateEuklideanDistanceBetweenMatrixes( meanOfFirstClass, meanOfSecondClass ) / ( covarianceMatrixOfFirstClass.det() + covarianceMatrixOfSecondClass.det() );
    }

        private double computeFisherLD( double[] vec, int [] ClassLabels, int [] SampleCount ) {
        // 1D, 2-classes
        double mA = 0, mB = 0, sA = 0, sB = 0;
        for ( int i = 0; i < vec.length; i++ ) {
            if ( ClassLabels[i] == 0 ) {
                mA += vec[i];
                sA += vec[i] * vec[i];
            } else {
                mB += vec[i];
                sB += vec[i] * vec[i];
            }
        }
        mA /= SampleCount[0];
        mB /= SampleCount[1];
        sA = sA / SampleCount[0] - mA * mA;
        sB = sB / SampleCount[1] - mB * mB;
        return Math.abs( mA - mB ) / ( Math.sqrt( sA ) + Math.sqrt( sB ) );
    }
                    

    private void calculateMeanValuesForEachClass( Map<String, double[][]> probesGroupedByClass ) {
        meanValuesPerClass = new HashMap<>();
        probesGroupedByClass.keySet().stream().forEach( ( className ) -> {
            double[] meanAttributeValues = calculateMeanValues( probesGroupedByClass.get( className ) );
            meanValuesPerClass.put( className, meanAttributeValues );
        } );

    }

    private double[] calculateMeanValues( double[][] probesInClass ) {
        double[] meanAttributeValues = new double[probesInClass.length];
        for ( int attributeIndex = 0; attributeIndex < probesInClass.length; attributeIndex++ ) {
            double sum = DoubleStream.of( probesInClass[attributeIndex] ).sum();
            meanAttributeValues[attributeIndex] = sum / probesInClass[attributeIndex].length;
        }
        return meanAttributeValues;
    }

    private void calculateCovarianceMatrixForEachClass( Map<String, double[][]> probesGroupedByClass, Map<String, double[]> meanValuesPerClass ) {
        covarianceMatrixesPerClass = new HashMap<>();
        probesGroupedByClass.keySet().stream().forEach( ( className ) -> {
            Matrix probesInClass = new Matrix( probesGroupedByClass.get( className ) );
            Matrix meanValuesInClass = new Matrix( meanValuesPerClass.get( className ), 1 );
            Matrix covarianceMatrixInClass = calculateCovarianceMatrix( probesInClass, meanValuesInClass );
            covarianceMatrixesPerClass.put( className, covarianceMatrixInClass );
        } );
    }

    private Matrix calculateCovarianceMatrix( Matrix probesInClass, Matrix meanValuesInClass ) {
        Matrix meanVectorAsMatrix = convertVectorToMatrix( probesInClass.getColumnDimension(), meanValuesInClass.getRowPackedCopy() );
        Matrix probesMinusMean = probesInClass.minus( meanVectorAsMatrix );
        Matrix covarianceMatrixInClass = probesMinusMean.times( probesMinusMean.transpose() );
        return covarianceMatrixInClass;
    }

    private double[][] combineProbesToOneMatrix( Map<String, double[][]> probesGroupedByClass ) {

        double[][] firstMatrix = new Matrix( probesGroupedByClass.get( classNames.get( 0 ) ) ).transpose().getArray();
        double[][] secondMatrix = new Matrix( probesGroupedByClass.get( classNames.get( 1 ) ) ).transpose().getArray();

        double result[][] = new double[firstMatrix.length + secondMatrix.length][firstMatrix[0].length];
        for ( int i = 0; i < firstMatrix.length; i++ ) {
            result[i] = firstMatrix[i].clone();
        }
        for ( int i = 0; i < secondMatrix.length; i++ ) {
            result[firstMatrix.length + i] = secondMatrix[i].clone();
        }

        return new Matrix( result ).transpose().getArray();
    }

    private Matrix convertVectorToMatrix( int requiredSize, double[] vectorMatrix ) {
        double[][] vectorAsMatrix = new double[requiredSize][vectorMatrix.length];
        for ( int index = 0; index < requiredSize; index++ ) {
            vectorAsMatrix[index] = vectorMatrix;
        }
        return new Matrix( vectorAsMatrix ).transpose();
    }

    private Matrix extractRequiredFeatures( double[] matrix, int[] requiredFeatures ) {
        double[] matrixWithOnlyRequiredFeatures = new double[requiredFeatures.length];
        int featureIndex = 0;
        for ( int requiredFeature : requiredFeatures ) {
            matrixWithOnlyRequiredFeatures[featureIndex] = matrix[requiredFeature];
            featureIndex++;
        }
        return new Matrix( matrixWithOnlyRequiredFeatures, 1 );
    }

    private Matrix extractRequiredFeatures( Matrix matrix, int[] requiredFeatures ) {
        double[][] matrixWithOnlyRequiredFeatures = new double[requiredFeatures.length][matrix.getColumnDimension()];
        int featureIndex = 0;
        for ( int requiredFeature : requiredFeatures ) {
            matrixWithOnlyRequiredFeatures[featureIndex] = matrix.getArray()[requiredFeature];
            featureIndex++;
        }
        return new Matrix( matrixWithOnlyRequiredFeatures );
    }

}
