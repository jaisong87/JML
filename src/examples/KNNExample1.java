package examples;
import classifiers.KNN;
import classifiers.NaiveBayes;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Vector;

import common.ClassifierException;
import common.FeatureVector;
import common.FeatureVectorException;
import common.TrainingException;


public class KNNExample1 {

	static int getPixelValueFromInput(FileReader fr) throws IOException{
		int ch = fr.read();
		if(ch ==10)
			ch=fr.read();
		
		if(ch == -1) 
			return -1; /*End of Input */
		
		if(ch == 35) /* Ascii for #  (black pixel)*/
			return 2;
		else if(ch == 43) /* Ascii for + (grey pixel) */
			return 1; /* ' ' white pixel */
		return 0;
	}

	static int getNextChar(FileReader fr) throws IOException{
		int ch = fr.read();
		if(ch ==10)
			ch=fr.read();
		return ch;
	}

	public static void main(String args[]) throws IOException, TrainingException, ClassifierException, FeatureVectorException{
		
		  FileReader trainImageStream = null, trainlabelStream = null, testImageStream = null, testLabelStream = null;
		  KNN knnClassifier = new KNN(3);
		  
		  try {
			  trainImageStream = new FileReader("trainingimages");
			  trainlabelStream = new FileReader("traininglabels");
			  
			  int digit, fval;
			  while( (digit = getNextChar(trainlabelStream)) != -1)
			  {
				  String classLabel = (char)digit+"";
				  Vector<Integer> numericValues = new Vector<Integer>();
				  for(int i=0;i<28;i++)
					  for(int j=0;j<28;j++)
					  {
						  fval = getPixelValueFromInput(trainImageStream);
						  numericValues.add(fval);
					  }
				  FeatureVector fVector = new FeatureVector( null, numericValues, null, classLabel, 0.0);
				  knnClassifier.addTrainingExample(fVector);
			  }
			  System.out.println("Successfully added trainingSet");

			  testImageStream = new FileReader("testimages");
			  testLabelStream = new FileReader("testlabels");
			  int testSamples = 0;
			  while( (digit = getNextChar(testLabelStream)) != -1)
			  {
				  testSamples++;
				  String classLabel = (char)digit+"";
				  Vector<Integer> numericValues = new Vector<Integer>();
				  for(int i=0;i<28;i++)
					  for(int j=0;j<28;j++)
					  {
						  fval = getPixelValueFromInput(testImageStream);
						  assert fval>=0 && fval<=2;
						  numericValues.add(fval);
					  }
				  FeatureVector fVector = new FeatureVector(null, numericValues, null, classLabel, 0.0);
				  knnClassifier.addTestExample(fVector);
			  }
			  System.out.println("Successfully added testSet");
		  }
		  finally {
			  System.out.println("Successfully read input");
		  }
		  
		  knnClassifier.printTrainingSet("trainingDump1","");
		  knnClassifier.printTestSet("testDump1","");
		  knnClassifier.train();
		  knnClassifier.test();
		  return;
	}
}
