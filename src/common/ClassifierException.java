package common;

public class ClassifierException extends Exception {

    String strValue;

	public ClassifierException( String value) {
         this.strValue = value;
	}

	public String toString() {
	   return "JML Exception : " + strValue;
        }
}