package examples;
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


public class NaiveBayesExample1 {

	static int getNextChar(FileReader fr) throws IOException{
		int ch = fr.read();
		if(ch ==10)
			ch=fr.read();
		return ch;
	}
	
	public static void main(String args[]) throws IOException, TrainingException, ClassifierException, FeatureVectorException{
		
		  FileReader trainImageStream = null, trainlabelStream = null, testImageStream = null, testLabelStream = null;
		  NaiveBayes nBayes = new NaiveBayes();
		  
		  try {
			  trainImageStream = new FileReader("trainingimages");
			  trainlabelStream = new FileReader("traininglabels");
			  
			  int digit;
			  char fval;
			  while( (digit = getNextChar(trainlabelStream)) != -1)
			  {
				  String classLabel = (char)digit+"";
				  Vector<String> nominalValues = new Vector<String>();
				  for(int i=0;i<28;i++)
					  for(int j=0;j<28;j++)
					  {
						  fval = (char)getNextChar(trainImageStream);
						  nominalValues.add(fval+"");
					  }
				  FeatureVector fVector = new FeatureVector( nominalValues, null, null, classLabel, 0.0);
				  nBayes.addTrainingExample(fVector);
			  }

			  testImageStream = new FileReader("testimages");
			  testLabelStream = new FileReader("testlabels");
			  while( (digit = getNextChar(testLabelStream)) != -1)
			  {
				  String classLabel = (char)digit+"";
				  Vector<String> nominalValues = new Vector<String>();
				  for(int i=0;i<28;i++)
					  for(int j=0;j<28;j++)
					  {
						  fval = (char)getNextChar(testImageStream);
						  nominalValues.add(fval+"");
					  }
				  FeatureVector fVector = new FeatureVector(nominalValues, null, null, classLabel, 0.0);
				  nBayes.addTestExample(fVector);
			  }
		  }
		  finally {
			  System.out.println("Successfully read input");
		  }
		  
		  nBayes.printTrainingSet("trainingDump1","");
		  nBayes.printTestSet("testDump1","");
		  nBayes.train();
		  nBayes.test();
		  return;
	}
}
