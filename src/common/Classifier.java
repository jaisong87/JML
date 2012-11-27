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
	String _classifierName;

	Classifier() {
		_trainingVector = new Vector<FeatureVector>();
		_testVector = new Vector<FeatureVector>();
		_classifierName = "Unknown Classifier";
	}

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

	public abstract boolean train();
	public abstract boolean test();
	
	public void printTrainingSet() {
		System.out.println("-------- Training set for : "+_classifierName+ "----------------");
		for(int i=0;i<_trainingVector.size();i++)
			System.out.println(_trainingVector.elementAt(i).getFeatureCSVString(","));	
		System.out.println("----------------------------------------------------------------");
		return;
	}
	
	public void printTestSet() {
		System.out.println("--------- Testing set for : "+_classifierName+ "----------------");
		for(int i=0;i<_testVector.size();i++)
			System.out.println(_testVector.elementAt(i).getFeatureCSVString(","));	
		System.out.println("----------------------------------------------------------------");
		return;
	}
}
