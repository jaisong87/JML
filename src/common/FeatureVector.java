/**
 * 
 */
package common;

import java.util.Vector;

/**
 * @author jaison
 *
 */
public class FeatureVector {
	
	private Vector<String> _categoricalValues;
	private Vector<Integer> _numericalValues;
	private Vector<Double> _realValues;

	String _class; /* Class of this data point */
	float _functionValue; /* Used for regression */
	
	FeatureVector(Vector<String> ctValues, Vector<Integer> nmValues, Vector<Double> rlValues, 
			String classification, float funcVal) {
		_categoricalValues = ctValues;
		_numericalValues = nmValues;
		_realValues = rlValues;
		
		_class = classification;
		_functionValue = funcVal;
	}

	
	/*A few checks */
	public boolean hasRealValues() {
		if(_realValues != null && _realValues.size()>0 )
			return true;
		return false;
	}
	
	public boolean hasCategoricalValues() {
		if(_categoricalValues != null && _categoricalValues.size()>0 )
			return true;
		return false;
	}

	public boolean hasNumericalValues() {
		if(_numericalValues != null && _numericalValues.size()>0 )
			return true;
		return false;
	}

	/* Get the above features */
	public Vector<String>  getCategoricalValues() {
		return _categoricalValues;
	}
	
	public Vector<Integer> getNumericalValues() {
		return _numericalValues;
	}
	
	public Vector<Double> getRealValues() {
		return _realValues;
	}
	
	/* Get the dimensionality of this featureVector */
	public int getSize() {
		int size = 0;
		if(_numericalValues != null )
			size += _numericalValues.size();
		if(_realValues != null)
			size += _realValues.size();
		if(_categoricalValues != null)
			size += _categoricalValues.size();
		return size;
	}
	
	/* Could be used to output CSVs etc*/
	public String getFeatureCSVString(String delim) {
		String featureStr = "";
		if(_numericalValues != null )
			{
				for(int i=0;i<_numericalValues.size();i++)
					featureStr+=_numericalValues.elementAt(i)+delim;
			}
		if(_realValues != null)
			{
			for(int i=0;i<_realValues.size();i++)
				featureStr+=_realValues.elementAt(i)+delim;
			}
		if(_categoricalValues != null)
			{
			for(int i=0;i<_categoricalValues.size();i++)
				featureStr+=_categoricalValues.elementAt(i)+delim;
			}
		
		if(_class != null)
			featureStr+=_class;
		else /*if(_functionValue != null)*/ 
			featureStr += _functionValue;
		/*else 
			featureStr += "<UNKNOWN_CLASS_OR_FUNCTIONVALUE";*/

		return featureStr;
	}

	public boolean isClassificationProblem() {
		if(_class==null)
			return false;
		return true;
	}
	
	/* Check whether two feature vectors are of same type */
	public boolean checkCompatibility(FeatureVector fv) {
		if( getSize() != fv.getSize())
			return false;
		
		if( (hasCategoricalValues()^(fv.hasCategoricalValues()) ) == true )
				return false;

		if( (hasRealValues()^(fv.hasRealValues()) ) == true )
			return false;

		if( (hasNumericalValues()^(fv.hasNumericalValues()) ) == true )
			return false;
				
		if( (isClassificationProblem()^(fv.isClassificationProblem()) ) == true )
			return false;
				
		return true;
	}
}
