/**
 * 
 */
package classifiers;

import java.util.HashMap;
import java.util.Vector;

import common.Classifier;
import common.ClassifierException;
import common.CommonUtils;
import common.FeatureVector;
import common.FeatureVectorException;
import common.TrainingException;

/**
 * @author jaison
 *
 */
public class ID3DecisionTree extends Classifier {

	/**
	 * 
	 */
	public ID3DecisionTree() {
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see common.Classifier#train()
	 */
	@Override
	public boolean train() throws TrainingException, ClassifierException {
		// TODO Auto-generated method stub
		return false;
	}

	/* (non-Javadoc)
	 * @see common.Classifier#predictClass(common.FeatureVector)
	 */
	@Override
	protected String predictClass(FeatureVector fv) throws TrainingException,
			ClassifierException, FeatureVectorException {
		// TODO Auto-generated method stub
		return null;
	}

	/*
	 * Calculate the info gain for splitting across a nominal 
	 */
	private double getInformationGainOnNominalFeature(Vector<FeatureVector> vf, int featureIdx) {
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
	private double getInformationGainOnRealFeature(Vector<FeatureVector> vf, int featureIdx, int splitPoint) {
		double rootEntropy = CommonUtils.getEntropy(vf);
		double childEntropy = 0;
		int totalSamples = vf.size();
		
		Vector<FeatureVector> leftSubTree = new Vector<FeatureVector>();
		Vector<FeatureVector> rightSubTree = new Vector<FeatureVector>();
		
		for(int i=0;i<totalSamples;i++) { /* Segragate into sets based on nominal feature value*/
			FeatureVector curSample = vf.get(i);
			double realValue = curSample.getRealValues().get(featureIdx);
			
			if(realValue < splitPoint)
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

}
