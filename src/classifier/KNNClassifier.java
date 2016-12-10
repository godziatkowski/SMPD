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

    /**
     * Metoda przyjmuje mape z parą kluczy dla każdej z klas np:
     * Alfa_training, Alfa_test, Beta_training, Beta_test
     * i rozdziela dane próbki na dwie mapy:
     * triningSets - Alfa_training, Beta_training
     * testSets - Alfa_test, Beta_test
     */
    private void splitPassedSetsIntoTestAndTraining( Map<String, double[][]> probesSplitedIntoTrainingAndTestSets ) {
        trainingSets = new HashMap<>();
        testSets = new HashMap<>();

        probesSplitedIntoTrainingAndTestSets.keySet() //z mapy pobieramy zestaw kluczy które posiada
                .forEach(className -> {// iteracja po kluczach w zestawie mapy
                    if (className.contains("_test")) { //jesli klucz posiada w sobie słowo "_test" oznacza to że jest to zbiór próbek testowych z danej klasy 
                        testSets.put(className, probesSplitedIntoTrainingAndTestSets.get(className));//zbiór próbek testowych odkładamy do mapy testSets po takim sammym kluczem jaki był w głównej mapie
                    } else {//jesli nazwa klasy nie zwiera dopiska _test oznacza to że jest to zbiór próbek treningowych i należy je odłożyć do zbioru treningowego
                        trainingSets.put(className, probesSplitedIntoTrainingAndTestSets.get(className));
                    }
                });
        getNamesOfTrainingAndTestSets(); // wywołanie metody do wydzielenia nazw klas 
    }

    @Override
    public double train( Map<String, double[][]> probesSplitedIntoTrainingAndTestSets, Set<Integer> indexesOfBestAttributes, int kCount ) {
        int countOfFailedAssignments = 0; // ilosc poprawnie zakwalifikowanych próbek
        int countOfSuccessAssignments = 0; // ilosc błędnie zakwalifikowanych próbek
        //obie powyższe wartości są wykorzystywane przy obliczeniu procentowej poprawności algorytmu
        splitPassedSetsIntoTestAndTraining( probesSplitedIntoTrainingAndTestSets ); // metoda do odzielenia próbek testowych od treningowych

        for ( String keyInTestSet : keysInTestSets ) { //dla kazdego klucza w mapie z probkami testowymi
            for ( int testProbeIndex = 0; testProbeIndex < testSets.get( keyInTestSet )[0].length; testProbeIndex++ ) { //iteracja po próbkach testowych
                int probeIndex = testProbeIndex;
                List<ProbeDistance> distancesBetweenProbeAndNearestClassNeighbour = new ArrayList<>(); // lista któa będzie przechowywać odległość pomięzy próbką testową, a każdą próbką z zestawu treningowego (nie zależnie od klasy)
                keysInTrainingSets.stream().forEach( ( keyInTrainingSet ) -> {//iteracja po kluczach w mapie z próbkami treningowymi
                    List<Double> closestClassNeighbours = new ArrayList<>(); // lista z najbliższymi próbkami z danej klasy próbek treningowych
                    for ( int trainingProbeIndex = 0; trainingProbeIndex < trainingSets.get( keyInTrainingSet )[0].length; trainingProbeIndex++ ) { // iteracja po próbkach treningowych z klasy
                        //wartość odległości Euklidesowej pomiędzy próbką testową a treningową
                        //na pdostawie wzoru: https://pl.wikipedia.org/wiki/Odleg%C5%82o%C5%9B%C4%87
                        double distanceToProbe = 0.0;
                        for ( int attributeIndex = 0; attributeIndex < trainingSets.get( keyInTrainingSet ).length; attributeIndex++ ) {//iteracja po cechach próbki
                            if ( indexesOfBestAttributes.isEmpty() || indexesOfBestAttributes.contains( attributeIndex ) ) { // jeśli kolekcja z najlepszymi cechami jest pusta (uwzględnienie wszystkich cech) lub kolekcja z najlepszymi cechami zawiera indeks obecnie iterowanej cechy
                                // obliczanie odległości pomiędzy próbka testowoą a treningową (pod względem danej cechy                                
                                // a = testSets.get( keyInTestSet )[attributeIndex][probeIndex] - pobranie wartości cechy spod attributeIndex dla próbki testowej na indeksie probeIndex
                                // b = trainingSets.get( keyInTrainingSet )[attributeIndex][trainingProbeIndex] - pobranie wartości cechy spod attributeIndex dla próbki treningowej na indeksie trainingProbeIndex                                
                                // odległość += (a-b)^2
                                distanceToProbe += Math.pow( testSets.get( keyInTestSet )[attributeIndex][probeIndex] - trainingSets.get( keyInTrainingSet )[attributeIndex][trainingProbeIndex], 2 );
                            }
                        }
                        distanceToProbe = Math.sqrt( distanceToProbe ); // w pętli powyżej robiliśmy kwadrat, teraz należało spierwiastkowanać wartość odległości
                        if ( closestClassNeighbours.size() < kCount ) { // jeśli ilość najbliższych próek z klasy nie przekracza podanej ilości przez paramtr ilości najbliższych próek
                            closestClassNeighbours.add( distanceToProbe );// to po prostu dodaj odległość jako najbliższą
                        } else {//w przeciwnym wypadku
                            Collections.sort( closestClassNeighbours );//posortuj odległości od najmniejszej do największej

                            if ( distanceToProbe < closestClassNeighbours.get( closestClassNeighbours.size() - 1 ) ) {// i sprawdz czy obliczona odległość jest mniejsza niż największa odległość na liscie najblizszych sasiadów
                                closestClassNeighbours.set( closestClassNeighbours.size() - 1, distanceToProbe );//jeśli tak to podmien ostatnia wartość
                            }
                        }
                    }
                    //zakonczenie iteracji po próbkach treningowych w danej klasie 
                    //poniższy kod opakowuje wyliczone najbliżśze odlegości w klasie do obiektu ProbeDistance w którym zawarta jest informacja o nazwie klasy oraz o odległości
                    List<ProbeDistance> smallestProbeDistances = closestClassNeighbours.stream()
                            .map( distance -> {
                                return new ProbeDistance( keyInTrainingSet, distance );
                            } ).collect( Collectors.toList() );

                    distancesBetweenProbeAndNearestClassNeighbour.addAll( smallestProbeDistances ); // dodanie obliczonych najbliższych odległości dla danej klasy do listy z najbliższymi odległościami nie uwzgledniającej klas
                } );

                String closestClass = "";
                Integer closestClassCount = null;
                Map<String, Integer> countOfClosestNegihboursPerClass = new HashMap<>();// Mapa <Nazwa_klasy, ilość_najbliżśzych_próbek>

                //mamy juz liste odleglosci pomiedzy probka testową, a (np. dziesięcioma dla kCount = 3) próbkami testowymi dla każdej z klas
                //teraz musimy je posortować po odległości bo:
                //jedna próbka z klasy Alfa może być odlegla o 10, druga o 25 trzecia o 30
                //natomiast z klasy Beta są odległości 15, 20, 27
                //chcemy miec 3 najbliższych sasiadów więc układamy próbki:
                //Alfa 10, Beta 15, Beta 20, Alfa 25, Beta 27, Alfa 30               
                Collections.sort( distancesBetweenProbeAndNearestClassNeighbour ); 
                //i dla tak uporządkowanych odległości sprawdzamy pierwsze 3 (kCount) próbki do jakiej klasy należą:
                //otrzymujemy Alfa - 1 próbka (10), oraz Beta - 2 próbki (15, 20)
                //jako że najbliższych próbek z klasy Beta jest więcej to stwierdzamy że badana próbka testowa należy do klasy Beta mimo że odległość do klasy najbliżśżej próbki z klasy Alfa jest mniejsza
                for ( int i = 0; i < kCount; i++ ) {//iteracja od 0 do kCount -1 ( x >= 0 i x < kCount)
                    ProbeDistance probeDistance = distancesBetweenProbeAndNearestClassNeighbour.get( i );//pobranie najbliżśzej próbki spod indeksu i
                    if ( countOfClosestNegihboursPerClass.containsKey( probeDistance.getClassName() ) ) {//sprawdzenie czy mapa countOfClosestNegihboursPerClass zawiera już klucz z nazwa klasy
                        countOfClosestNegihboursPerClass.put( probeDistance.getClassName(), ( countOfClosestNegihboursPerClass.get( probeDistance.getClassName() ) + 1 ) ); // jeśli tak to zwiększamy ilość najbliższych próbek o jeden
                    } else {//jesli nie
                        countOfClosestNegihboursPerClass.put( probeDistance.getClassName(), 1 );// wstaw pod nazwa klasy wartość 1 (jest jedna najbliżśza próbka z danej klasy)
                    }//nastepnie idz dalej do kolejnej próbki
                }

                //iteracja po kluczach (nazwach klas) w mapie countOfClosestNegihboursPerClass w celu znalezienia klasy która ma najwięcej najbliższych próbek względem badanej próbki
                for ( String keyInCountOfClosestNegihboursPerClass : countOfClosestNegihboursPerClass.keySet() ) {
                    // jeśli ilość najbliższych próbek nie została jesxcze ustalona (pierwsza iteracja w pętli) lub ilość najbliżśzych próek w obecnej iteracji jest wieksza niż poprzednia najwieksza wartość
                    if ( closestClassCount == null || closestClassCount < countOfClosestNegihboursPerClass.get( keyInCountOfClosestNegihboursPerClass ) ) {
                        //jeśli tak to podmieniamy nazwe najbliższej klasy 
                        closestClass = keyInCountOfClosestNegihboursPerClass;
                        //i zmieniamy ilość najbliszych próbek
                        closestClassCount = countOfClosestNegihboursPerClass.get( keyInCountOfClosestNegihboursPerClass );
                    }
                    //jesli nie idziemy dalej
                }

                //test czy próbka została poprawnie zakwalifikowana
                //jesli po ucięciu dopiska training z nazwy najbliższej klasy jej nazwa zawiera się w nazwie klasy do któej powinna należeć próbka testowa
                //to kwalifikacja jest poprawna i zwiększamy ilość sukcesów o 1
                //jeśli kwalifikacja była błędna to zwiększamy ilość niepoprawnych kwalifikacji
                //Czyli:
                //próbka testowa siedzi w mapie pod kluczem Alfa_test
                //i próbka ta zostałą zakwalifikowana do klasy Alfa_training
                //to ucinamy "_training" (Alfa_training ---> Alfa)
                //i sprawdzamy czy "Alfa_test".containg("Alfa") - true
                // jeśli byłoby Beta_training
                //to Beta_training bez dopiska ---> "Beta"
                //"Alfa_test".contains("Beta") = false                
                if ( keyInTestSet.contains( closestClass.replace( "_training", "" ).trim() ) ) {
                    countOfSuccessAssignments++;
                } else {
                    countOfFailedAssignments++;
                }
            }
        }
        //obliczenie procentowej poprawności algorytmu:
        // poprawność = a * 100 / (a+b) 
        //gdzie a to ilość poprawnych kwalifikacji
        //b to ilosc błędnych kwalifikacji
        // a+b w sumie daja ogólną ilość kwalifikacji
        double percentageAlgorithmCorrectness = ( (double) countOfSuccessAssignments / (double) ( countOfFailedAssignments + countOfSuccessAssignments ) ) * 100;
        return percentageAlgorithmCorrectness;
    }

}
