package classifier;

public class ProbeDistance implements Comparable<ProbeDistance> {

    private final String className;

    private final double distanceToTestProbe;
    
    /**
     * Obiekt opakowujący dla dwóch danych:
     * @param className - nazwa klasy do której należy próbka
     * @param distanceToTestProbe - odległość próbki z danej klasy od próbki testowej (któej klasa nie jest wiadoma)
     * Obiekt wykorzystywany w klasyfikatorze KNN
     */
    public ProbeDistance( String className, double distanceToTestProbe ) {
        this.className = className;
        this.distanceToTestProbe = distanceToTestProbe;
    }

    public String getClassName() {
        return className;
    }

    public double getDistanceToTestProbe() {
        return distanceToTestProbe;
    }

    @Override
    public int compareTo( ProbeDistance comparedObject ) {
        return Double.compare( this.distanceToTestProbe, comparedObject.getDistanceToTestProbe() );
    }

}
