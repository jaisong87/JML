/**
 * 
 */
package common;

import java.util.Vector;

/**
 * @author jaison
 *
 */
public abstract class Classifier {
	
	private Vector<FeatureVector> _trainingVector, _testVector;

	/* Add another training example */
	public boolean addTrainingExample(FeatureVector fv) {
		if(_trainingVector.size() > 0)
		{
			if(fv.checkCompatibility(_trainingVector.elementAt(0)) == false )
			{
				//add appropriate exception here
				return false;
			}	
		}
		
		_trainingVector.add(fv); /* This is compatible. so add it*/
		return true;
	}

	/* Add another training example */
	public boolean addTestExample(FeatureVector fv) {
		if(_testVector.size() > 0)
		{
			if(fv.checkCompatibility(_testVector.elementAt(0)) == false )
			{
				//add appropriate exception here
				return false;
			}	
		}
		
		_testVector.add(fv); /* This is compatible. so add it*/
		return true;
	}

}
