package examples;

import java.util.Vector;

import common.ClassifierException;
import common.FeatureVector;
import common.TrainingException;

import classifiers.ID3DecisionTree;

public class ID3DecisionTreeExample1 {

	public static void main(String args[]) throws TrainingException, ClassifierException {
		
		ID3DecisionTree dTree = new ID3DecisionTree();
		
		for(int i=0;i<10;i++) {
		
		Vector<String> nomValues = new Vector<String>();
		Vector<Double> realValues = new Vector<Double>();
		String classLabel;
		
		if(i%2==0) {
			classLabel = "Male";
			nomValues.add("Rough");
			realValues.add(20.0+(i/10));
		}
		else {
			classLabel = "FeMale";
			nomValues.add("Soft");			
			realValues.add(10.0+(i/10));
		}

		FeatureVector fv = new FeatureVector(null, null, realValues, classLabel, 0);
		dTree.addTrainingExample(fv);
		}
		
		dTree.train();
		System.out.println("Successfully run ID3 on the given data");
		dTree.displayDebugInfo();
	}
}
