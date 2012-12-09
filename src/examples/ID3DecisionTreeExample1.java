package examples;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.Vector;

import common.ClassifierException;
import common.CommonUtils;
import common.CommonUtils.EntropyMeasure;
import common.FeatureVector;
import common.FeatureVectorException;
import common.TrainingException;

import classifiers.ID3DecisionTree;

public class ID3DecisionTreeExample1 {

	public static Double parseDouble(String d) {
		Double val = null;
		if(!d.equals("?"))
			val = Double.parseDouble(d);
		
		return val;
	}
	
	public static void main(String args[]) throws TrainingException, ClassifierException, IOException, FeatureVectorException {
		
		ID3DecisionTree dTree = new ID3DecisionTree();
		
			
			  FileInputStream inputStream = new FileInputStream("samples");
			  BufferedReader br = new BufferedReader(new InputStreamReader(inputStream, Charset.forName("UTF-8")));
			  String line;
			  int sampleCount = 0;

			  Vector<FeatureVector> allSamples = new Vector<FeatureVector>();
			  
			  while( ((line = br.readLine()) != null ) )
			  {
				  sampleCount++;
				  //System.out.println("Line : "+line);
				  String[] values = line.split(",");
				  
				  Vector<String> nomValues = new Vector<String>();
				  nomValues.add(values[0]);
				  nomValues.add(values[3]);
				  nomValues.add(values[4]);
				  nomValues.add(values[5]);
				  nomValues.add(values[6]);
				  nomValues.add(values[8]);
				  nomValues.add(values[9]);
				  nomValues.add(values[11]);
				  nomValues.add(values[12]);
				  
				  Vector<Double> realValues = new Vector<Double>();
				  realValues.add(parseDouble(values[1]));
				  realValues.add(parseDouble(values[2]));
				  realValues.add(parseDouble(values[7]));
				  realValues.add(parseDouble(values[10]));
				  realValues.add(parseDouble(values[13]));
				  realValues.add(parseDouble(values[14]));
				  
				  String classLabel = values[15];				  
				  FeatureVector fv = new FeatureVector(nomValues, null, realValues, classLabel, 0.0);
				  allSamples.add(fv);
			  }

			  dTree.setEntropyMeasure(EntropyMeasure.GAIN_RATIO);

			  double trainingSplit = 0.80;
			  int sampleSize = allSamples.size();

			  System.out.print(CommonUtils.getCompleteFeatureProfile(allSamples));
			  System.out.println("EntropyMeasure "+dTree.getEntropyMeasure());
			  
			  for(int i=0;i<(trainingSplit*sampleSize);i++)
				  dTree.addTrainingExample(allSamples.get(i));

			  System.out.println("Successfully added trainingSet");
			  
			  for(int i=(int)(trainingSplit*sampleSize);i<sampleSize;i++)
				  dTree.addTestExample(allSamples.get(i));

			  System.out.println("Successfully added testSet");

			  dTree.train();

			  	System.out.println("Successfully run ID3 on the given data");
				dTree.displayDebugInfo();

			  dTree.test();
			  
	}
}
