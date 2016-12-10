package classifier;

import Jama.Matrix;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static probesUtilities.MatrixUtilities.getProbe;
import static probesUtilities.MatrixUtilities.pseudoinverseMoorePenrose;

public class NMClassifier implements IClassifier {

    private Map<String, double[][]> trainingSets;
    private Map<String, double[][]> testSets;
    private Set<String> keysInTrainingSets;
    private Set<String> keysInTestSets;

    private void getNamesOfTrainingAndTestSets() {
        keysInTestSets = testSets.keySet();
        keysInTrainingSets = trainingSets.keySet();
    }

    /**
     * Metoda opisana w klasie KNNClassifier
     * @param probesSplitedIntoTrainingAndTestSets 
     */
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

    /**
     * Metoda do wyliczenia średniej wartości klas
     * np:
     * próbka składa się z trzech atrybutów [X,Y,Z]
     * i mamy w klase Alfa próbki: a[2,5,8], b[1,4,9] i c[3,6,4]
     * z tych trzech próbek chcemy zrobić uśrednioną pojedynczą próbke:
     * d[Xd, Yd, Zd]
     * gdzie:
     * Xd = (Xa + Xb + Xc) / n = (2 + 1 + 3) / 3 = 6/3 = 2 (n --> 3 jako że sa 3 próbki)
     * i analitycznie dla Yd i Zd
     * Yd = (5 + 4 + 6) / 3 = 5
     * Zd = (8 + 9 + 4) / 3 = 7
     * d[2,5,7] - jest to próbka wyznaczająca srednia wartość klasy
     * @param trainingSets
     * @return 
     */
    private Map<String, double[]> calculateMeanValues( Map<String, double[][]> trainingSets ) {

        Map<String, double[]> meanAttributesPerClass = new HashMap<>();
        trainingSets.entrySet()
                .forEach( entry -> { // dla każdego elementu w mapie gdzie element = Klucz, Wartość
                    double[] means = new double[entry.getValue().length];
                    for ( int attributeIndex = 0; attributeIndex < entry.getValue().length; attributeIndex++ ) {//iteracja po atrybutach
                        double mean = 0.0;
                        //petla sumująca wartości dla danej cechy
                        for ( int probeIndex = 0; probeIndex < entry.getValue()[attributeIndex].length; probeIndex++ ) {
                            mean += entry.getValue()[attributeIndex][probeIndex]; 
                        }
                        mean = mean / entry.getValue()[attributeIndex].length; // dzielenie zsumowanej wartosci cechy przez ilosc probek (srednia :D )
                        means[attributeIndex] = mean;
                    }
                    meanAttributesPerClass.put( entry.getKey(), means );// odłożenie obliczonej wartosci sredniej dla klasy do mapy <Klasa, WartośćŚrednia>
                } );

        return meanAttributesPerClass;
    }
    /**
     * metoda do wybrania określonych cech próek
     * Jesli mamy probki o cechach probka[A,B,C,D,E,F]
     * ale cechy B i E są beznadziejne i ich porównywanie zabiera jedynie czas
     * to eliminujemy je (wyznaczenie które cechy są ok jest gdzie indziej - tutaj jedynie przyjmujemy paramtr z indeksami najleszpych cech i te cechy zostawiamy)
     * [A,B,C,D,E,F] --> [A,C,D,F]
     * @param probesWithAllAttributes - próbki testowe, treningowe, dla wszystkich klas
     * @param indexesOfBestAttributes - lista cech które są OK (może to być tylko jedna cecha, może być ich więcej)
     * @return
     */
    private Map<String, double[][]> extractSelectedAttributes( Map<String, double[][]> probesWithAllAttributes, List<Integer> indexesOfBestAttributes ) {
        Map<String, double[][]> probesWithExtractedAttributes = new HashMap<>();
        probesWithAllAttributes.keySet().stream().forEach( ( className ) -> { //iteracja po nazwach klas (Alfa_training, Alfa_test itp itd)
            double[][] probesInSet = probesWithAllAttributes.get( className ); // próbki w klasie
            double[][] probesInSetWithExtractedAttributes = new double[indexesOfBestAttributes.size()][probesInSet[0].length]; // zdefiniowane nowej tablicy do trzymanai próbek z ograniczona liczbą cech
            int index = 0;
            for ( Integer indexOfBestAttribute : indexesOfBestAttributes ) {//kopiowanie danej cechy dla wszystkich próbek
                probesInSetWithExtractedAttributes[index] = probesInSet[indexOfBestAttribute];
                index++;
            }
            probesWithExtractedAttributes.put( className, probesInSetWithExtractedAttributes );//odlozenie próbek (z wyeliminowanymi cechami) do mapy pod takim samym kluczem jak wczesniej
        } );

        return probesWithExtractedAttributes;
    }
    /**
     * Metoda obliczająca maceirz kowariancji
     * http://www.itl.nist.gov/div898/handbook/pmc/section5/pmc541.htm
     * próbka sklada się z 3 elementów [x,y,z]
     * i mamy pięć próbek
     * A=[4.0, 2.0, 0.60]
     * B=[4.2, 2.1, 0.59]
     * C=[3.9, 2.0, 0.58]
     * D=[4.3, 2.1, 0.62]
     * E=[4.1, 2.2, 0.63]
     * 
     * Z tego mamy wartosci srednie:
     * SR = [4.1, 2.08, 0.604]
     * to macierz kowariancji jest rozmiarów 3x3 (bo mamy 3 atrybuty)
     *  MK =
     * [ mk1, mk2, mk4 ]
     * [ mk2, mk3, mk5 ]
     * [ mk4, mk5, mk6 ]
     * 
     * gdzie obliczanie wartości polega na:
     * mk1 = (( ( A[x] - SR[x] )*( A[x] - SR[x] ) ) + ( ( B[x] - SR[x] )*( B[x] - SR[x] ) ) + ( ( C[x] - SR[x] )*( C[x] - SR[x] ) ) + ( ( D[x] - SR[x] )*( D[x] - SR[x] ) ) + ( ( E[x] - SR[x] )*( E[x] - SR[x] ) )) / n-1  <--- m = liczna probek w tym wypadku 5
     * mk1 = (( (4.0 - 4.1) * (4.0 - 4.1) ) + ( (4.2 - 4.1) * (4.2 - 4.1) ) + ( (3.9 - 4.1) * (3.9 - 4.1) ) + ( (4.3 - 4.1) * (4.3 - 4.1) ) + ( (4.1 - 4.1) * (4.1 - 4.1) )) / 4
     * mk1 = (( (-0.1) * (-0.1) ) + ( 0.1 * 0.1 ) + ( (-0.2) * (-0.2) ) + ( 0.2 * 0.2 ) +( 0 * 0 )) /4
     * mk1 = (0.01 + 0.01 + 0.04 + 0.04 + 0) / 4
     * mk1 = 0.1 /4 = 0.025
     * analogicznie wylicza sie mk3 i mk6 (elementy na przekatnej
     * Natomiast elementy poza przekatna liczy sie:
     * mk2 = (( ( A[x] - SR[x] )*( A[y] - SR[y] ) ) + ( ( B[x] - SR[x] )*( B[y] - SR[y] ) ) + ( ( C[x] - SR[x] )*( C[y] - SR[y] ) ) + ( ( D[x] - SR[x] )*( D[y] - SR[y] ) ) + ( ( E[x] - SR[x] )*( E[y] - SR[y] ) )) / 4
     * róznica mk1 lezy w wierszu 1 w kolumnie 1 (lub na odwrot)
     * natomiast mk2 - wiersz = 1, kolumna = 2 lub wiersz = 2 i kolumna = 1
     * w pojedynczym wyrazie wiec (przed sumowaniem):
     * ( A[x] - SR[x] )*( A[x] - SR[x] ) zamienia sie w ( A[x] - SR[x] )*( A[y] - SR[y] )
     * czyli: 
     * A[x] - SR[x] - różnica wartosci cechy próbki A dla cechy x - mk1 lezy w [1,1] więc powatarzamy ten sam wyraz 
     * mk2 lezy w [1,2] / [2,1] więc musimy połączyć różnice dla pierwszej i drugiej cechy (cecha x i y)
     * stad:
     * (A[x] - SR[x] )*( A[y] - SR[y])
     * gdzie A[x] - SR[x] - różnica miedzy probka a srednia dla cechy x
     * A[y] - SR[y] - róznica miedzy próbka a srednia dla cechy y
     */
    private Map<String, double[][]> calculateCovarianceMatrix( Map<String, double[][]> probesWithSelectedAttributes, Map<String, double[]> meanValuesForEachClass ) {
        Map<String, double[][]> covarianceMatrixesForEachClass = new HashMap<>();

        probesWithSelectedAttributes.keySet().stream().forEach( className -> { // obliczanie macierzy kowariancji dla kazdej klasy
            double[][] probesInClass = probesWithSelectedAttributes.get( className );
            int countOfAttributes = probesInClass.length;
            int countOfProbes = probesInClass[0].length;
            double[][] covarianceMatrix = new double[countOfAttributes][countOfAttributes];
            double[] meanValuesForEachAttribute = meanValuesForEachClass.get( className );
            //iteracje po cechach - dla cech [x,y,z] musimy przejsc po kazdej parze: 
            //[x,x], [x,y], [x,z]
            //[y,x], [y,y], [y,z]
            //[z,x], [z,y], [z,z]
            for ( int firstAttributeIterator = 0; firstAttributeIterator < countOfAttributes; firstAttributeIterator++ ) { // iteracja po pierwszym indeksie w parze
                for ( int secondAttributeIterator = 0; secondAttributeIterator < countOfAttributes; secondAttributeIterator++ ) { //iteracja po drugim indeksie w parze
                    double covarianceValue = 0.0; // obliczanie wartosci poszeczgólnych elementów w macierzy kowariancji
                    for ( int probeIndex = 0; probeIndex < countOfProbes; probeIndex++ ) { //iteracja po probkach (jest tu założenie ze kazda probka ma wszystkie cechy)
                        //sumowanie dla kazdej próbki                       
                        covarianceValue += ( probesInClass[firstAttributeIterator][probeIndex] - meanValuesForEachAttribute[firstAttributeIterator] ) * ( probesInClass[secondAttributeIterator][probeIndex] - meanValuesForEachAttribute[secondAttributeIterator] );
                    }
                    
                    covarianceValue = covarianceValue / ( countOfProbes - 1 );
                    covarianceMatrix[firstAttributeIterator][secondAttributeIterator] = covarianceValue;
                }
            }
            Matrix matrix = new Matrix( covarianceMatrix ); // obiekt z Jamy reprezentujacy macierz
            try { // próba odwrócenia macierzy 
                matrix = matrix.inverse();// jesli wyznacznik macierzy = 0 to rzucany jest wyjatek, nie oznacza to jednak ze odwrócenie macierzy jest niemożliwe
            } catch ( Exception e ) {
                matrix = pseudoinverseMoorePenrose( matrix );// wywołanie odwracania macierzy metoda Moore Penrose - omija wyjatek dla wyznacznika = 0
            }
            covarianceMatrixesForEachClass.put( className, matrix.getArray() );
        } );

        return covarianceMatrixesForEachClass;
    }

    @Override
    public double train( Map<String, double[][]> probesSplitedIntoTrainingAndTestSets, Set<Integer> indexesOfBestAttributes ) {
        //usuniećie "zbędnych" cech z próek - zostają tylko cechy przekazane w indexesOfBestAttributes
        Map<String, double[][]> probesSplitedIntoTrainingAndTestSetsWithExtreactedAttributes = extractSelectedAttributes( probesSplitedIntoTrainingAndTestSets, indexesOfBestAttributes.stream().collect( Collectors.toList() ) );
        // rozdzielenie probek z zestawó testowych i treningowych na osobne mapy
        splitPassedSetsIntoTestAndTraining( probesSplitedIntoTrainingAndTestSetsWithExtreactedAttributes ); 
        double percentageAlgorithmCorrectness = 0.0;
        int countOfFailedClassifications = 0;
        int countOfSuccessClassifications = 0;

        Map<String, double[]> meanAttributesPerClass = calculateMeanValues( trainingSets );//obliczenie srednich wartosci (wektorów średnich) 
        Map<String, double[][]> covarianceMatrixes = calculateCovarianceMatrix( trainingSets, meanAttributesPerClass ); // obliczenie macierzy kowariancji

        for ( String keyInTestSet : keysInTestSets ) { // iteracja po zbiorach testowych
            double[][] probesInTestSet = testSets.get( keyInTestSet );
            for ( int probeIndexInTestSet = 0; probeIndexInTestSet < probesInTestSet[0].length; probeIndexInTestSet++ ) { // iteracja po probkach w danym zbiorze testowym
                Map<String, Double> distanceBetweenProbeAndCLassCentroid = new HashMap<>();
                double[] probe = getProbe( probesInTestSet, probeIndexInTestSet );
                // konwersja próbki z dostaci double[] na obiekt macierzy (w zasadzie na wektor ale kazdy wketor jest macierza)
                Matrix probeAsMatrix = new Matrix( probe, 1 ); 
                // iteracja po zbiorach treningowych
                keysInTrainingSets.stream().forEach( ( keyInTrainingSet ) -> { 
                    // wyciagniecie macierzy kowariancji dla danego zbioru treningowego
                    Matrix covarianceMatrix = new Matrix( covarianceMatrixes.get( keyInTrainingSet ) ); 
                    // wyciagniecie wektora srednich dla danego zbioru treningowego
                    Matrix meanMatrix = new Matrix( meanAttributesPerClass.get( keyInTrainingSet ), 1 ); 
                    // odejmowanie macierzy (wektorów) 
                    // jesli wektor V1 = [a,b,c] wektor V2 = [d,e,f]
                    // to wektor V3 = V1 - V2 = [ a-d, b-e, c-f ]
                    Matrix probeMinusMean = probeAsMatrix.minus( meanMatrix ); 
                    //mnozenie macierzy obliczonej powyzej (wektora) przez macierz kowariacnji i przez macierz transoponowana wzgledem obliczonej powyzej
                    // M = V3 * CovMat * transpose(V3)
                    // M - macierz wynikowa,
                    // V3 - wektor (macierz) obliczona  lini 210
                    // CovMat - macierz kowariancji
                    // transponse(V3) - macierz V3 transoponowana ( jesli były 3 elementy w wierszu to teraz sa te same 3 elementy tylko ze w kolumnie)
                    //na indeksie [0,0] macierzy znajduje się odległość pomiedzy testowana próbka a klasa (dlatego na koncu jest get(0,0)
                    double distanceToSet = probeMinusMean.times( covarianceMatrix ).times( probeMinusMean.transpose() ).get( 0, 0 );
                    //odlozenie odleglosci probka-klasa do mapy
                    distanceBetweenProbeAndCLassCentroid.put( keyInTrainingSet, distanceToSet );
                } );
                double smallestDistance = 1000000.0; // w zasadzie powinno byc Double.MAX_VALUE - jest to po prostu bardzo duza liczba która powinna być wieksza niz jakakolwiek odleglosc probka-klasa
                String closestClass = "";
                Set<String> keySet = distanceBetweenProbeAndCLassCentroid.keySet();
                //wyznacznie najblizszej klasy:
                //iteracja po nazwach klas (kluczach w mapie)
                for ( String key : keySet ) {
                    //sprawdzenie czy odleglosc od obecnie iterowanej klasy jest mniejsza niz poprzednia wartosc smallestDistance
                    if ( distanceBetweenProbeAndCLassCentroid.get( key ) < smallestDistance ) {
                        //jesli tak to podmieniamy najmniejsza wartosc
                        smallestDistance = distanceBetweenProbeAndCLassCentroid.get( key );
                        //i zmieniamy nazwe klasy na obecnie iterowana
                        closestClass = key;
                    }
                }
                //sprawdzenie czy probka była poprawnie zakwalifikowana (po wiecej wyjasnien patrz KNNClassifier)
                if ( keyInTestSet.contains( closestClass.replace( "_training", "" ).trim() ) ) {
                    countOfSuccessClassifications++;
                } else {
                    countOfFailedClassifications++;
                }
            }
        }
        //obliczenie procentowej poprawnosci algorytmu
        percentageAlgorithmCorrectness = ( (double) countOfSuccessClassifications / (double) ( countOfFailedClassifications + countOfSuccessClassifications ) ) * 100;

        return percentageAlgorithmCorrectness;
    }

}
