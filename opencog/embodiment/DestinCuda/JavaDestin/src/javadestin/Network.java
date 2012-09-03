package javadestin;

import java.util.ArrayList;
import java.util.List;

public class Network {
	public static class NetworkConfig {
		// TODO: is this a good default learning rate?
		public float fixedLearningRate = .001f;
	}

	private LayerFinishedCallback callback;
	private List<DestinKernel> layers = new ArrayList<DestinKernel>();
	private RunningInfo ri = new RunningInfo();
	// how many times doDestin is called
	private int callcount = 0;

	public void free(){
		if(layers!=null){
			int n = layers.size();
			for(int l = 0 ; l < n ; l++){
				layers.get(l).delete(); //call delete and call destructor on native DestinKernel to free graphics memory.
			}
			layers = null;
		}
	}
	public Network(int[] centroidCounts, int pixelsPerInputNode,
			NetworkConfig config) {
		creatStandard(centroidCounts, pixelsPerInputNode, config);
	}

	private void creatStandard(int[] centroidCounts,
			int pixelsPerInputNode, NetworkConfig config) {
		int nlayers = centroidCounts.length;

		DestinKernel layer;
		CurandGeneratorWrapper gen = new CurandGeneratorWrapper();

		for (int l = 0; l < nlayers; l++) {
			layer = new DestinKernel();
			int width = (int) Math.pow(2, nlayers - l - 1);
			int inputDim = l == 0 ? pixelsPerInputNode
					: centroidCounts[l - 1] * 4;
			int parentStates = l == nlayers - 1 ? 1 : centroidCounts[l + 1];
			// TODO: is ID set right?
			layer.Create(l, width, width, centroidCounts[l], parentStates,
					inputDim, config.fixedLearningRate, gen.getReference());

			if (l > 0) {
				layers.get(l - 1).SetInputAdvice(layer.GetOutputAdvice());
			}
			if (l == (nlayers - 1)) {
				layer.SetInputAdvice(null);
			}
			layers.add(layer);
		}
	}

	public void setIsPOSTraining(boolean training){
		for (int i = 0; i < layers.size(); i++) {
			layers.get(i).setIsPOSTraining(training);
		}		
	}
	public void setIsPSSATraining(boolean training){
		for (int i = 0; i < layers.size(); i++) {
			layers.get(i).setIsPSSATraining(training);
		}
	}

	public void doDestin(SWIGTYPE_p_float dInput) {
		if (dInput == null) {
			throw new RuntimeException("dInput cannot be null");
		}
		DestinKernel layer;
		ri.setImage_count(callcount);
		for (int l = layers.size() - 1; l >= 0; l--) {
			ri.setLayer(l);
			layer = layers.get(l);

			if (l == 0) {
				layer.DoDestin(dInput, null);
			} else {
				layer.DoDestin(layers.get(l - 1).GetDevicePointerBeliefs(),
						null);
			}
			if (callback != null) {
				this.callback.callback(ri, layer);
			}
		}
		callcount++;
	}

	public void setLayerFinishedCallback(LayerFinishedCallback callback) {
		this.callback = callback;
	}

}
