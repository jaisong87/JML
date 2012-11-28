package classifiers;

import java.util.HashMap;

import common.Classifier;
import common.ClassifierException;
import common.CommonUtils;
import common.FeatureVector;
import common.FeatureVectorException;
import common.TrainingException;

public class KNN extends Classifier {

	private int _k;
	
	KNN() {
		_k = -1; /* User should explicitly set K*/
	}

	public void setK(int K) {
		_k = K;
	}

	
	@Override
	public boolean train() throws TrainingException, ClassifierException{
		/* Nothing here currently - can make a Kd-Tree or an R* Tree for better indexing and lookup */
		if(_k <= 0)
			throw new ClassifierException("For KNNClassifier , setK should be called before train()");
			
		return false;
	}
	
	@Override
	protected String predictClass(FeatureVector fv) throws ClassifierException, FeatureVectorException {
		if(_k <= 0)
			throw new ClassifierException("For KNNClassifier , parameter K should be set before running the model");

		if(fv.getCategoricalValues()!=null)
			throw new ClassifierException("KNNClassifier cannot handle Feature vectors with nominal values");

		double[] distances = new double[_k];
		String[] labels     = new String[_k]; 
		
		for(int i=0;i<_trainingVector.size();i++)
		{
			FeatureVector curExample = _trainingVector.elementAt(i);
			double euclidianDis = CommonUtils.getEuclidianDistance(curExample, fv);
			String curLabel = curExample.getClassification();
			
			if(i< _k)
			{
				distances[i] = euclidianDis;
				labels[i] = curLabel;
			}
			else {
				/* If euclidianDis < KthClosestDataPoint*/
				if(euclidianDis < distances[_k-1])
				{
					int pos = 0;
					while(distances[pos] < euclidianDis ) pos++;
					
					for(int j=_k-1;j>pos;j--)
					{
						distances[j] = distances[j-1];
						labels[j] = labels[j-1];
					}
					
					distances[pos] = euclidianDis;
					labels[pos] = curLabel;
				}
			}
		}

		String bestClass = null;
		int maxVotes = 0;
		HashMap<String, Integer> majorityVoter = new HashMap<String, Integer>();
		for(int i=0;i<_k;i++)
		{
			Integer frq = majorityVoter.get(labels[i]);
			int votes = 0;
			if(frq != null) 
					votes = frq + 1;
			
			majorityVoter.put(labels[i], votes);
			
			if(bestClass == null) {
				bestClass = labels[i];
				maxVotes = votes;
			}
			else if(votes> maxVotes){
				maxVotes = votes;
				bestClass = labels[i];
			}
		}
		return bestClass;
	}

	
}
