package common;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Vector;

public class CommonUtils {

	public enum EntropyMeasure { INFO_GAIN, GAIN_RATIO };
	public enum DistanceMeasure { MAHATTAN_DIS, EUCLIDIAN_DIS };
	
	public static String getMajorityVote(Vector<FeatureVector> vf) {
		HashMap<String, Integer> classFrqMap = new HashMap<String, Integer>();
		int totalSamples = vf.size();
		
		for(int i=0;i<totalSamples;i++){
			String curClass = vf.get(i).getClassification();	
			Integer curFrq = classFrqMap.get(curClass);
			if(curFrq != null)
				classFrqMap.put(curClass, curFrq + 1);
			else classFrqMap.put(curClass, 1);
		}
		
		int bestFrq = 0;
		String bestClass ="";
		
		for(Map.Entry<String, Integer> classFrq : classFrqMap.entrySet()){
			Integer frq = classFrq.getValue();
			
			if(frq > bestFrq) {
				bestFrq = frq;
				bestClass = classFrq.getKey();
			}
		}
		return bestClass;
	}
	
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
		double intrinsicValue = 0;
		int totalSamples = vf.size();
		
		Vector<FeatureVector> leftSubTree = new Vector<FeatureVector>();
		Vector<FeatureVector> rightSubTree = new Vector<FeatureVector>();
		Vector<FeatureVector> unknownSubTree = new Vector<FeatureVector>();
		
		for(int i=0;i<vf.size();i++) { /* Segragate into sets based on nominal feature value*/
			FeatureVector curSample = vf.get(i);
			Double realValue = curSample.getRealValues().get(featureIdx);
			
			if(realValue != null) {
			if(realValue < curSplitPoint)
				leftSubTree.add(curSample);
			else 	
				rightSubTree.add(curSample);
		} else {
			unknownSubTree.add(curSample);
		}
			}

		if(leftSubTree.size() > rightSubTree.size()) 
		{
			leftSubTree.addAll(unknownSubTree);
		}
		else {
			rightSubTree.addAll(unknownSubTree);				
		}
		
		/* calculate child entropy as weighted sum of entropies across left and right children */
		 
			double leftTreeWeight = (double)leftSubTree.size()/(double)totalSamples;
			intrinsicValue += leftTreeWeight*CommonUtils.getEntropy(leftSubTree);

			double rightTreeWeight = (double)rightSubTree.size()/(double)totalSamples;
			intrinsicValue += rightTreeWeight*CommonUtils.getEntropy(rightSubTree);
			
		return rootEntropy - intrinsicValue;
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
	
	public static String getRealFeatureProfile(Vector<FeatureVector> vf, int featureIdx) {
		String profileInfo = "=========== Profiling r"+featureIdx + " =============== \n";
		
		HashMap<Double, Integer> values = new HashMap<Double, Integer>();
		double sum = 0, mean = 0, median = 0, mode =0;
		double min = vf.get(0).getRealValues().get(featureIdx);
		double max = vf.get(0).getRealValues().get(featureIdx);

		int totSamples = vf.size(), maxFrq = 0, unknownValues = 0;

		for(int i=0;i<totSamples;i++)
		{
			Double curVal = vf.get(i).getRealValues().get(featureIdx);
			
			if(curVal !=null) {
			Integer curValFrq = values.get(curVal);
			if(curValFrq == null)
				curValFrq = 1;
			
			values.put(curVal , curValFrq);
			if( curValFrq > maxFrq)
			{
				maxFrq = curValFrq;
				mode = curVal;
			}

			if(curVal < min) min = curVal;
			if(curVal > max) max = curVal;
			
			sum +=curVal;
			}
			else {
				unknownValues++;
			}
		}
		
		mean = sum/totSamples;
		profileInfo += (" Min      : " + min  + "\n");
		profileInfo += (" Max      : " + max  + "\n" );
		profileInfo += (" Mean     : " + mean + "\n" );
		profileInfo += (" Mode     : " + mode + "\n");
		profileInfo += (" Unknown  : " + unknownValues + "\n");
		profileInfo += (" Distinct : " + values.size() +"\n");
		profileInfo += (" Values : ");
		
		int totValues = values.size(), curIdx = 0 ;

//		Iterator itr = values.entrySet().iterator();

		for(Map.Entry<Double,Integer> entry : values.entrySet() ) {
				double curValue = (double) entry.getKey();
				curIdx++;
				if( totValues <= 20)
					profileInfo += (curValue+",");
				else if(curIdx <7 || curIdx>=(totValues-7))
					profileInfo += (curValue+",");
				else if(curIdx == 7 )
					profileInfo += (".............");
		}
			profileInfo += ("\n");
		
		return profileInfo;
	}

	public static String getClassLabelProfile(Vector<FeatureVector> vf) {
		HashMap<String, Integer> classLabelProfile = new HashMap<String, Integer>();
		
		int totSamples = vf.size();
		for(int i=0;i<totSamples;i++) {
			String cLabel = vf.get(i).getClassification();
			Integer frq = classLabelProfile.get(cLabel);
			if(frq == null)
				frq = 0;
			
			classLabelProfile.put(cLabel, frq + 1);
		}
		
		String cLabelInfo = " ++++ CLASSES : ";
		for(Map.Entry<String, Integer> cInfo: classLabelProfile.entrySet())
			{
				double weight = (double)cInfo.getValue()/totSamples;
				cLabelInfo += "( " + cInfo.getKey() + "["+ cInfo.getValue()+"]" + weight +" ) ";
			}
		return cLabelInfo + "++++++\n";
	} 
	
	public static String getCompleteFeatureProfile(Vector<FeatureVector> vf) {
		FeatureVector fv = vf.get(0);
		String featureInfo = "--------- Profile info for FeatureVector "+vf +"("+vf.size()+") samples --------\n";
		int realFeatureCount = (fv.hasRealValues())?fv.getRealValues().size() : 0;
		int nomFeatureCount = (fv.hasCategoricalValues())?fv.getCategoricalValues().size():0;
		int numFeatureCount = (fv.hasNumericalValues())?fv.getNumericalValues().size():0;
		
		
		featureInfo += "  Total Features : "+(realFeatureCount+nomFeatureCount+numFeatureCount)+"\n";
		featureInfo += "   Real Features : "+(realFeatureCount)+"\n";
		featureInfo += "Nominal Features : "+(nomFeatureCount)+"\n";
		featureInfo += "Numeric Features : "+(numFeatureCount)+"\n";
		
		for(int i=0;i<realFeatureCount;i++)
			featureInfo += getRealFeatureProfile(vf, i);
		
		if(fv.isClassificationProblem())
				featureInfo += getClassLabelProfile(vf);
		
		return featureInfo;
	}
}
