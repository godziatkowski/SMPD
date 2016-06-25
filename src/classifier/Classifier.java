package classifier;

public class Classifier {

    double[][] TrainingSet, TestSet;
    int[] ClassLabels;
    final int TRAIN_SET = 0, TEST_SET = 1;

    public void generateTraining_and_Test_Sets( double[][] Dataset, String TrainSetSize ) {

        int[] Index = new int[Dataset[0].length];
        double Th = Double.parseDouble( TrainSetSize ) / 100.0;
        int TrainCount = 0, TestCount = 0;
        for ( int i = 0; i < Dataset[0].length; i++ ) {
            if ( Math.random() <= Th ) {
                Index[i] = TRAIN_SET;
                TrainCount++;
            } else {
                Index[i] = TEST_SET;
                TestCount++;
            }
        }
        TrainingSet = new double[Dataset.length][TrainCount];
        TestSet = new double[Dataset.length][TestCount];
        TrainCount = 0;
        TestCount = 0;
        // label vectors for training/test sets
        for ( int i = 0; i < Index.length; i++ ) {
            if ( Index[i] == TRAIN_SET ) {
                System.arraycopy( Dataset[i], 0, TrainingSet[TrainCount++], 0, Dataset[0].length );
            } else {
                System.arraycopy( Dataset[i], 0, TestSet[TestCount++], 0, Dataset[0].length );
            }
        }
    }

    public void trainClissifier( double[][] TrainSet ) {

    }

}
