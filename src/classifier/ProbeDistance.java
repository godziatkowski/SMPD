package classifier;

public class ProbeDistance implements Comparable<ProbeDistance> {

    private final String className;

    private final double distanceToTestProbe;

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
