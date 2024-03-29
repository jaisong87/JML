package classifiers;

import java.util.HashMap;
import java.util.Vector;

import common.Classifier;
import common.FeatureVector;

public class NaiveBayes extends Classifier {

	private HashMap<String, Integer> _classFrequency;    /* <K, V> => < Class, Count(Class) >*/

	/* likelihoodFrequnecyForNums[C][featureIdx][featureVal] = Count(Class=C, va for featureIdx = featureVal)*/
	private HashMap<String, HashMap<Integer, HashMap<Integer, Integer >>> _likelihoodFrequnecyForNums;

	/* likelihoodFrequnecyForDouble[C][featureIdx][featureVal] = Count(Class=C, va for featureIdx = featureVal)*/
	private HashMap<String, HashMap<Integer, HashMap<Double, Integer >>> _likelihoodFrequnecyForDouble;
	
	/* likelihoodFrequnecyForNominals[C][featureIdx][featureVal] = Count(Class=C, va for featureIdx = featureVal)*/
	private HashMap<String, HashMap<Integer, HashMap<String, Integer >>> _likelihoodFrequnecyForNominals;
	
	public NaiveBayes() {
		// TODO Auto-generated constructor stub
		_classifierName = "NaiveBayes";
		
		/**
		 * Posterior = (prior * likelihood )/evidence
		 */
		_classFrequency = new HashMap<String, Integer>();
		_likelihoodFrequnecyForNums = new HashMap<String, HashMap<Integer, HashMap<Integer, Integer >>>();
		_likelihoodFrequnecyForDouble = new HashMap<String, HashMap<Integer, HashMap<Double, Integer >>>();
		_likelihoodFrequnecyForNominals = new HashMap<String, HashMap<Integer, HashMap<String, Integer >>>();
	}

	/**
	 * @param args
	 */
/*	public static void main(String[] args) {
		
		NaiveBayes nBayes = new NaiveBayes();
		
		// TODO Auto-generated method stub
		for(int i=0;i<40;i++)
		{
			Vector<Integer> numValues = new Vector<Integer>();
			for(int j=0;j<4;j++)
			{
				numValues.add(i+j);
			}
			String classification = "1";
			if(i<10)  classification = "1";
			else if(i<=20) classification = "2";
			else if(i<=30) classification = "3";
			else classification = "4";
			
			FeatureVector fv = new FeatureVector(null, numValues, null, classification, 0.0 );
			nBayes.addTrainingExample(fv);
		}
		
		nBayes.printTrainingSet();
	}*/

	/* Add a data point Frequency(Class=Ci, Feature[featureIdx] = value) */
	private void addRealDataPoint(String classification, int featureIdx, Double value) {
		HashMap<Integer, HashMap<Double, Integer>> classMap = _likelihoodFrequnecyForDouble.get(classification);
		if(classMap == null)
		{
			classMap = new HashMap<Integer, HashMap<Double, Integer >>();
			_likelihoodFrequnecyForDouble.put(classification, classMap);
		}
		
		HashMap<Double, Integer> frqCount = classMap.get(featureIdx);
		if(frqCount == null )
		{
			frqCount = new HashMap<Double, Integer>();
			classMap.put(featureIdx, frqCount);
		}
		
		Integer curFrqCount = frqCount.get(value);
		if(curFrqCount == null)
		{
			frqCount.put(value, 1);
		}
		else {
			frqCount.put(value, 1 + curFrqCount);
		}
	}

	/* Add a data point Frequency(Class=Ci, Feature[featureIdx] = value) */
	private void addNumericDataPoint(String classification, int featureIdx, Integer value) {
		HashMap<Integer, HashMap<Integer, Integer>> classMap = _likelihoodFrequnecyForNums.get(classification);
		if(classMap == null)
		{
			classMap = new HashMap<Integer, HashMap<Integer, Integer >>();
			_likelihoodFrequnecyForNums.put(classification, classMap);
		}
		
		HashMap<Integer, Integer> frqCount = classMap.get(featureIdx);
		if(frqCount == null )
		{
			frqCount = new HashMap<Integer, Integer>();
			classMap.put(featureIdx, frqCount);
		}
		
		Integer curFrqCount = frqCount.get(value);
		if(curFrqCount == null)
		{
			frqCount.put(value, 1);
		}
		else {
			frqCount.put(value, 1 + curFrqCount);
		}
	}

	/* Add a data point Frequency(Class=Ci, Feature[featureIdx] = value) */
	private void addNominalDataPoint(String classification, int featureIdx, String value) {
		HashMap<Integer, HashMap<String, Integer>> classMap = _likelihoodFrequnecyForNominals.get(classification);
		if(classMap == null)
		{
			classMap = new HashMap<Integer, HashMap<String, Integer >>();
			_likelihoodFrequnecyForNominals.put(classification, classMap);
		}
		
		HashMap<String, Integer> frqCount = classMap.get(featureIdx);
		if(frqCount == null )
		{
			frqCount = new HashMap<String, Integer>();
			classMap.put(featureIdx, frqCount);
		}
		
		Integer curFrqCount = frqCount.get(value);
		if(curFrqCount == null)
		{
			frqCount.put(value, 1);
		}
		else {
			frqCount.put(value, 1 + curFrqCount);
		}
	}

	/* Get Frequency(Class=Ci, Feature[featureIdx] = value) */
	private double getRealDataLikelihood(String classification, int featureIdx, Double value) {
		HashMap<Integer, HashMap<Double, Integer>> classMap = _likelihoodFrequnecyForDouble.get(classification);
		if(classMap == null)
		{
			classMap = new HashMap<Integer, HashMap<Double, Integer >>();
			_likelihoodFrequnecyForDouble.put(classification, classMap);
		}
		
		HashMap<Double, Integer> frqCount = classMap.get(featureIdx);
		if(frqCount == null )
		{
			frqCount = new HashMap<Double, Integer>();
			classMap.put(featureIdx, frqCount);
		}
		
		Integer curFrqCount = frqCount.get(value);
		if(curFrqCount == null)
		{
			return 0.0;
		}
		else {
			int classFrequency = _classFrequency.get(classification);
			int keySize = _classFrequency.keySet().size();
			double likelihood = (1.0 + curFrqCount)/(keySize + classFrequency );
			assert likelihood<=1.0;
			return likelihood;
		}
	}

	/* Get data point Frequency(Class=Ci, Feature[featureIdx] = value) */
	private double getNumericDataLikelihood(String classification, int featureIdx, Integer value) {
		HashMap<Integer, HashMap<Integer, Integer>> classMap = _likelihoodFrequnecyForNums.get(classification);
		if(classMap == null)
		{
			classMap = new HashMap<Integer, HashMap<Integer, Integer >>();
			_likelihoodFrequnecyForNums.put(classification, classMap);
		}
		
		HashMap<Integer, Integer> frqCount = classMap.get(featureIdx);
		if(frqCount == null )
		{
			frqCount = new HashMap<Integer, Integer>();
			classMap.put(featureIdx, frqCount);
		}
		
		Integer curFrqCount = frqCount.get(value);
		if(curFrqCount == null)
		{
			return 0.0;
		}
		else {
			int classFrequency = _classFrequency.get(classification);
			int keySize = _classFrequency.keySet().size();
			double likelihood = (1.0 + curFrqCount)/(keySize + classFrequency );
			assert likelihood<=1.0;
			return likelihood;
		}
	}

	/* Get Frequency(Class=Ci, Feature[featureIdx] = value) */
	private double getNominalDataLikelihood(String classification, int featureIdx, String value) {
		HashMap<Integer, HashMap<String, Integer>> classMap = _likelihoodFrequnecyForNominals.get(classification);
		if(classMap == null)
		{
			classMap = new HashMap<Integer, HashMap<String, Integer >>();
			_likelihoodFrequnecyForNominals.put(classification, classMap);
		}
		
		HashMap<String, Integer> frqCount = classMap.get(featureIdx);
		if(frqCount == null )
		{
			frqCount = new HashMap<String, Integer>();
			classMap.put(featureIdx, frqCount);
		}

		Integer curFrqCount = frqCount.get(value);
		if(curFrqCount == null)
		{
			return 0.0;
		}
		else {
			int classFrequency = _classFrequency.get(classification);
			int keySize = _classFrequency.keySet().size();
			double likelihood = (1.0 + curFrqCount)/(keySize + classFrequency );
			assert likelihood<=1.0;
			return likelihood;
		}
	}

	private double getPriorProbability(String classification) {
		int totalTrainingSamples = _trainingVector.size();
		int classFrequency = _classFrequency.get(classification);
		double prior = (1.0*classFrequency)/totalTrainingSamples;
		assert prior<=1.0;
		return prior;
	}
	
	/* Predict probability of a class - THIS NEEDS LAPLACE SMOOTHENING */
	private double predictProbability(String classification, FeatureVector fv){
		double posterior = 0.0;
		
		/* Multiply with prior*/
		double prior = getPriorProbability(classification);
		posterior += Math.log(prior);
		
		/* Now multiply with likelihood */
		Vector<Double> realValues = fv.getRealValues();
		if(realValues != null) {
		for(int i=0;i<realValues.size();i++)
			{
				posterior += Math.log(getRealDataLikelihood(classification, i, realValues.elementAt(i)));
			}	
		}

		Vector<Integer> numericValues = fv.getNumericalValues();
		if(numericValues != null) {
			for(int i=0;i<numericValues.size();i++)
			{
				posterior += Math.log(getNumericDataLikelihood(classification, i, numericValues.elementAt(i)));
			}	
		}

		Vector<String> nomValues = fv.getCategoricalValues();
		if(nomValues != null){
			for(int i=0;i<nomValues.size();i++)
			{
				posterior += Math.log(getNominalDataLikelihood(classification, i, nomValues.elementAt(i)));
			}	
		}

		return posterior;
	}
	
	protected String predictClass(FeatureVector fv) {
		String bestClass = null;
		double curBest = 0.0;
		
		for(String classLabel:_classFrequency.keySet())
		{
			double curPosterior = predictProbability(classLabel, fv);
			System.out.print("["+curPosterior+"("+classLabel+"),");
			if(bestClass == null ){
				bestClass = classLabel;
				curBest = curPosterior;
			}
			else if( curBest < curPosterior)
			{
				bestClass = classLabel;
				curBest = curPosterior;				
			}
		}
		System.out.println(" => "+fv.getClassification()+"<-->"+bestClass);
		
		return bestClass;
	}
	
	@Override
	public boolean train() {
		
		if(_trainingVector.size() == 0 )
			return false;
		
		/* Get all the classes with frequencies so that we have the priors */
		for(int i=0;i<_trainingVector.size();i++)
		{
			String currentClass = _trainingVector.elementAt(i).getClassification();
			final Integer frq = _classFrequency.get(currentClass);
			if(frq != null )
			{
				_classFrequency.put(currentClass, frq + 1);
			}
			else {
				_classFrequency.put(currentClass, 1);
			}
		}
		
		if(_classFrequency.size() <= 1) /* Should contain at least two classes for classification */
			return false;
		
		
		for(int i=0;i<_trainingVector.size();i++)
		{
			FeatureVector curTrainigVector = _trainingVector.elementAt(i);
			String classification = curTrainigVector.getClassification();
			
			/* Update real data point for class classification*/
			Vector<Double> realValues = curTrainigVector.getRealValues();
			if(realValues != null)
				{
				for(int j=0;j<realValues.size();j++)
				addRealDataPoint(classification, j, realValues.elementAt(j));
				}
			
			/* Update numeric data point for class classification */
			Vector<Integer> numValues = curTrainigVector.getNumericalValues();
			if(numValues != null) {
			for(int j=0;j<numValues.size();j++)
				this.addNumericDataPoint(classification, j, numValues.elementAt(j));
				}
			
			/* Update nominal data point for class classification */
			Vector<String> nomValues = curTrainigVector.getCategoricalValues();
			if(nomValues != null) {
			for(int j=0;j<nomValues.size();j++)
				addNominalDataPoint(classification, j, nomValues.elementAt(j));
				}
			}
		return true;
	}

}
