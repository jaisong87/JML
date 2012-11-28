/**
 * 
 */
package common;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Formatter;
import java.util.HashSet;
import java.util.Set;
import java.util.Vector;
import java.util.HashMap;


/**
 * @author jaison
 *
 */
public abstract class Classifier {
	protected Vector<FeatureVector> _trainingVector;
	protected Vector<FeatureVector> _testVector;
	protected String _classifierName;
	protected HashMap<String, HashMap<String, Integer> > _confusionMatrix;
	
	public Classifier() {
		_trainingVector = new Vector<FeatureVector>();
		_testVector = new Vector<FeatureVector>();
		_classifierName = "Unknown Classifier";
		_confusionMatrix = new HashMap<String, HashMap<String, Integer> >();
	}

	/* clear confusion matrix */
	public void clearConfusionMatrix(){
		if(_confusionMatrix != null)
			_confusionMatrix.clear();
	}
	
	/* Add to confusion Matirx*/
	public void addResultToConfusionMAtrix(String origClass, String predClass){
		if(_confusionMatrix == null)
			_confusionMatrix = new HashMap<String, HashMap<String, Integer> >();
		HashMap<String, Integer> confTable = _confusionMatrix.get(origClass);
		
		if(confTable == null)
		{
			confTable = new HashMap<String, Integer>();
			_confusionMatrix.put(origClass, confTable);
		}
		
		Integer frq = confTable.get(predClass);
		if(frq == null)
			frq = 0;
		confTable.put(predClass, frq + 1);
		return;
	}
	
	protected int getPredictionCount(String origClass, String predClass) {
		if(_confusionMatrix == null)
			return 0;

		HashMap<String, Integer> confTable = _confusionMatrix.get(origClass);
		if(confTable == null)
			return 0;
		
		Integer frq = confTable.get(predClass);
		if(frq == null)
			return 0;
		
		return frq;
	}
	
	
	void printConfusionMatrix()
	{
		Set<String> classLabels = _confusionMatrix.keySet();
		System.out.print("     |");
		HashSet<String> predLabels = new HashSet<String>();

		for(String cLabel : classLabels) {
			System.out.print(String.format("%3s  |", cLabel));
			predLabels.add(cLabel);
			}
		System.out.println();

		for(int i=0;i<=predLabels.size();i++)
			System.out.print("------");
		System.out.println();

		for(String origLabel : classLabels) {
			System.out.print(String.format("%3s  |", origLabel));
				for(String cLabel : predLabels)
				{
					int predCount = getPredictionCount(origLabel, cLabel);
					System.out.print(String.format("%3d  |", predCount));
				}
			System.out.println();
		}		
	}

	/* Add another training example */
	public void addTrainingExample(FeatureVector fv) throws TrainingException{
		if(_testVector.size() > 0)
		{
			if(fv.checkCompatibility(_testVector.elementAt(0)) == false )
				throw new TrainingException("Feature Vector not compatible " +
						"with previously added feature vectors in testSet");
		}
		
		if(_trainingVector.size() > 0)
		{
			if(fv.checkCompatibility(_trainingVector.elementAt(0)) == false )
			{
				throw new TrainingException("Feature Vector not compatible " +
						"with previously added feature vectors in trainingSet");
			}	
		}

		_trainingVector.add(fv); /* This is compatible. so add it*/
	}

	public boolean test() throws TrainingException, ClassifierException, FeatureVectorException {
		// TODO Auto-generated method stub
		clearConfusionMatrix();
		
		double samples = 0;
		double correct = 0;
		for(int i=0;i<_testVector.size();i++) {
			FeatureVector fv = _testVector.elementAt(i);
			String predictedLabel = predictClass(fv);
			String trueLabel = fv.getClassification();
			if(predictedLabel.equals(trueLabel))
				correct++;
			System.out.println(predictedLabel + " predicted for "+trueLabel);
			samples++;
			addResultToConfusionMAtrix(trueLabel, predictedLabel);
		}

		double accuracy = correct/samples;
		System.out.println("Accuracy : "+(100*accuracy)+"%");
		printConfusionMatrix();
		return false;
	}

	/* Add another training example */
	public void addTestExample(FeatureVector fv) throws TrainingException {
		if(_testVector.size() > 0)
		{
			if(fv.checkCompatibility(_testVector.elementAt(0)) == false )
				throw new TrainingException("Feature Vector not compatible " +
						"with previously added feature vectors in testSet");
		}
		
		if(_trainingVector.size() > 0)
		{
			if(fv.checkCompatibility(_trainingVector.elementAt(0)) == false )
			{
				throw new TrainingException("Feature Vector not compatible " +
						"with previously added feature vectors in trainingSet");
			}	
		}
		
		_testVector.add(fv); /* This is compatible. so add it*/
	}

	public abstract boolean train() throws TrainingException, ClassifierException;
	protected abstract String predictClass(FeatureVector fv) throws TrainingException, ClassifierException, FeatureVectorException;
	
	public void printTrainingSet(String fileName, String delim) throws IOException {
		
	if(fileName != null) {	
	    PrintWriter out = new PrintWriter(new FileWriter(fileName));	      
		out.println("-------- Training set for : "+_classifierName+ "("+_trainingVector.size()+" samples )----------------");
		for(int i=0;i<_trainingVector.size();i++)
			out.println(_trainingVector.elementAt(i).getFeatureCSVString(delim));	
		out.println("----------------------------------------------------------------"); 
		out.close();
	}
	else {
		System.out.println("-------- Training set for : "+_classifierName+ "----------------");
		for(int i=0;i<_trainingVector.size();i++)
			System.out.println(_trainingVector.elementAt(i).getFeatureCSVString(delim));	
		System.out.println("----------------------------------------------------------------"); 		
	}
		return;
	}

	public void printTestSet(String fileName, String delim) throws IOException {
		
	if(fileName != null) {	
	    PrintWriter out = new PrintWriter(new FileWriter(fileName));	      
		out.println("-------- Test set for : "+_classifierName+ "("+_testVector.size()+" samples )----------------");
		for(int i=0;i<_testVector.size();i++)
			out.println(_testVector.elementAt(i).getFeatureCSVString(delim));	
		out.println("----------------------------------------------------------------"); 
		out.close();
	}
	else {
		System.out.println("-------- Test set for : "+_classifierName+ "----------------");
		for(int i=0;i<_testVector.size();i++)
			System.out.println(_testVector.elementAt(i).getFeatureCSVString(delim));	
		System.out.println("----------------------------------------------------------------"); 		
	}
		return;
	}

}
