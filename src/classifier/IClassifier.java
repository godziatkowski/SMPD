package classifier;

import java.util.Map;
import java.util.Set;

//Interfejs dla klasyfikatorów NN, NM oraz KNM
public interface IClassifier {
    
    /**
     * metoda służąca do przydzielenia próbek do odpowiednich klas
     * @param probesSplitedIntoTrainingAndTestSets - mapa której kluczem jest nazwa klasy do której próbka jest zakwalifikowana / powinna być zakwalifikowana 
     * np dla klas Alfa i Beta:
     * pod kluczem Alfa_training / Beta_training znajdują się próbki należące do zbioru danych - wiemy że są to próbki alfa
     * natomiast pod kluczem Alfa_test / Beta_test, znajdują się próbki które mają być przetestować poprawność algorytmu segregującego 
     * algorytm klasyfikuje próbki z dopiskiem _test do grupy Alfa lub Beta po czym sprawdza czy próbka należała do grupy Alfa_test lub Beta_test
     * Jeśli próbka była podana pod kluczem Alfa_test i zostałą zakwalifikowana do klasy Alfa - próbka zakwalifikowana poprawnie
     * @param indexesOfBestAttributes
     * Zbiór niepowatarzających się cech które zostały oznaczone jako najlepsze (których użycie zwróci najbardziej prawidłowe wyniki)
     * @return algorytm zwraca procentową poprawność algorytmu - np dla zbioru 10 próbek testowych 8 zostało zakwalifikowanych poprawnie to zostanie zwrócona wartość 80.0
     */
    double train( Map<String, double[][]> probesSplitedIntoTrainingAndTestSets, Set<Integer> indexesOfBestAttributes );
    
}
