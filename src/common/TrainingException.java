package common;

public class TrainingException extends Exception {

    String strValue;

	public TrainingException( String value) {
         this.strValue = value;
	}

	public String toString() {
	   return "JML Exception : " + strValue;
        }
}