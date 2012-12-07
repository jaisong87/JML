package common;

import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

public class CommonUtils {

	public static double getEntropy(Vector<FeatureVector> vf) {
		
		if(vf.size() == 0)
			return 0.0; /* No entropy in an empty set */

		double entropy = 0;
		
		HashMap<String, Integer> classFrqMap = new HashMap<String, Integer>();
		int totalSamples = vf.size();
		
		for(int i=0;i<totalSamples;i++){
			String curClass = vf.get(i).getClassification();	
			Integer curFrq = classFrqMap.get(curClass);
			if(curFrq != null)
				classFrqMap.put(curClass, curFrq + 1);
			else classFrqMap.put(curClass, 1);
		}
		
		for(Map.Entry<String, Integer> classFrq : classFrqMap.entrySet()){
			Integer frq = classFrq.getValue();
			
			double classProbability = (double)frq/(double)totalSamples;
			entropy += (-1*classProbability*Math.log(classProbability)); /* Won't call log on 0 */
		}
		
		return entropy;
	}
	
	/*
	 * Calculate the info gain for splitting across a nominal 
	 */
	public static double getInformationGainOnNominalFeature(Vector<FeatureVector> vf, int featureIdx) {
		double rootEntropy = CommonUtils.getEntropy(vf);
		double childEntropy = 0;
		int totalSamples = vf.size();
		
		HashMap<String, Vector<FeatureVector>> childMap = new HashMap<String, Vector<FeatureVector>>();
		
		for(int i=0;i<totalSamples;i++) { /* Segragate into sets based on nominal feature value*/
			FeatureVector curSample = vf.get(i);
			String nominalValue = curSample.getCategoricalValues().get(featureIdx);
			
			Vector<FeatureVector> childVector = childMap.get(nominalValue);
			if(childVector == null)
				childVector = new Vector<FeatureVector>();
			
			childVector.add(curSample); 
			childMap.put(nominalValue, childVector);
		}
		
		/* calculate child entropy as weighted sum 
		 * of entropies across all children
		 */
		for(Vector<FeatureVector> childNode : childMap.values()) { 
			double childNodeWeight = (double)childNode.size()/(double)totalSamples;
			childEntropy += childNodeWeight*CommonUtils.getEntropy(childNode);
		}
		
		return rootEntropy - childEntropy;
	}

	/*
	 * Calculate the info gain for splitting across a real valued feature 
	 */
	public static double getInformationGainOnRealFeature(Vector<FeatureVector> vf, int featureIdx, double curSplitPoint) {
		double rootEntropy = CommonUtils.getEntropy(vf);
		double childEntropy = 0;
		int totalSamples = vf.size();
		
		Vector<FeatureVector> leftSubTree = new Vector<FeatureVector>();
		Vector<FeatureVector> rightSubTree = new Vector<FeatureVector>();
		
		for(int i=0;i<totalSamples;i++) { /* Segragate into sets based on nominal feature value*/
			FeatureVector curSample = vf.get(i);
			double realValue = curSample.getRealValues().get(featureIdx);
			
			if(realValue < curSplitPoint)
				leftSubTree.add(curSample);
			else 	
				rightSubTree.add(curSample);
		}
		
		/* calculate child entropy as weighted sum of entropies across left and right children */
		 
			double leftTreeWeight = (double)leftSubTree.size()/(double)totalSamples;
			childEntropy += leftTreeWeight*CommonUtils.getEntropy(leftSubTree);

			double rightTreeWeight = (double)rightSubTree.size()/(double)totalSamples;
			childEntropy += rightTreeWeight*CommonUtils.getEntropy(rightSubTree);
			
		return rootEntropy - childEntropy;
	}
	
	public static double getEuclidianDistance(FeatureVector f1, FeatureVector f2) 
			throws FeatureVectorException {
		double euclidianDis = 0;
		
		if(f1.checkCompatibility(f2) == false)
			throw new FeatureVectorException("FeatureVectors passed are " +
					"incompatible with each other for distance measurement");
		
		if(f1.hasCategoricalValues())
			throw new FeatureVectorException("Euclidian Distance " +
					"cannot be computed on feature Vectors with nominal values");

		if(f1.hasRealValues())
		{
			Vector<Double> realValues1 = f1.getRealValues();
			Vector<Double> realValues2 = f2.getRealValues();
		
			for(int i=0;i<realValues1.size();i++)
			{
				double d1 = realValues1.elementAt(i);
				double d2 = realValues2.elementAt(i);
				euclidianDis += Math.sqrt((d1-d2)*(d1-d2));
			}
		}
		
		if(f1.hasNumericalValues())
		{
			Vector<Integer> numValues1 = f1.getNumericalValues();
			Vector<Integer> numValues2 = f2.getNumericalValues();
		
			for(int i=0;i<numValues1.size();i++)
			{
				double d1 = numValues1.elementAt(i);
				double d2 = numValues2.elementAt(i);
				euclidianDis += Math.sqrt((d1-d2)*(d1-d2));
			}
		}
		
		return euclidianDis;
	}
}
