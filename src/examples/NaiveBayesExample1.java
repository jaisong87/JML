package examples;
import classifiers.NaiveBayes;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Vector;

import common.FeatureVector;


public class NaiveBayesExample1 {

	static int getNextChar(FileReader fr) throws IOException{
		int ch = fr.read();
		if(ch ==10)
			ch=fr.read();
		return ch;
	}
	
	public static void main(String args[]) throws IOException{
		//testimages
		//testlabels
		//trainingimages
		//traininglabels
		
		  FileReader trainImageStream = null, trainlabelStream = null, testImageStream = null, testLabelStream = null;
		  NaiveBayes nBayes = new NaiveBayes();
		  
		  try {
			  trainImageStream = new FileReader("trainingimages");
			  trainlabelStream = new FileReader("traininglabels");
			  
			  int digit, fval;
			  while( (digit = getNextChar(trainlabelStream)) != -1)
			  {
				  String classLabel = (char)digit+"";
				  Vector<Integer> numericFeatureValues = new Vector<Integer>();
				  for(int i=0;i<28;i++)
					  for(int j=0;j<28;j++)
					  {
						  fval = getNextChar(trainImageStream);
						  numericFeatureValues.add(fval);
					  }
				  FeatureVector fVector = new FeatureVector(null, numericFeatureValues, null, classLabel, 0.0);
				  nBayes.addTrainingExample(fVector);
			  }

			  testImageStream = new FileReader("testimages");
			  testLabelStream = new FileReader("testlabels");
			  while( (digit = getNextChar(testLabelStream)) != -1)
			  {
				  String classLabel = (char)digit+"";
				  Vector<Integer> numericFeatureValues = new Vector<Integer>();
				  for(int i=0;i<28;i++)
					  for(int j=0;j<28;j++)
					  {
						  fval = getNextChar(testImageStream);
						  numericFeatureValues.add(fval);
					  }
				  FeatureVector fVector = new FeatureVector(null, numericFeatureValues, null, classLabel, 0.0);
				  nBayes.addTestExample(fVector);
			  }
		  }
		  finally {
			  System.out.println("Successfully read input");
		  }
		  
		  nBayes.printTrainingSet("trainingDump1");
		  nBayes.train();
		  nBayes.test();
		  return;
	}
}
