package common;

public class FeatureVectorException extends Exception {

    String strValue;

	public FeatureVectorException( String value) {
         this.strValue = value;
	}

	public String toString() {
	   return "JML FeatureVectorException : " + strValue;
        }
}