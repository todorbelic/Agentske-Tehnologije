package training

import (
	messages "agentske/proto"
	"fmt"
	"github.com/asynkron/protoactor-go/actor"
	"gonum.org/v1/gonum/mat"
	"time"
)

func (n *MLP) Evaluate(x, y mat.Matrix) (float64, float64) {

	p := n.Predict(x)
	N, _ := p.Dims()

	var (
		truePositive  int
		falsePositive int
		falseNegative int
	)

	for n := 0; n < N; n++ {
		ry := mat.Row(nil, n, y)
		truth := ry[0]

		rp := mat.Row(nil, n, p)
		predicted := Prediction(rp)

		if predicted == 1.0 && truth == 1.0 {
			truePositive++
		} else if predicted == 1.0 && truth == 0.0 {
			falsePositive++
		} else if predicted == 0.0 && truth == 1.0 {
			falseNegative++
		}
	}
	precision := float64(truePositive) / float64(truePositive+falsePositive)
	recall := float64(truePositive) / float64(truePositive+falseNegative)
	if (precision + recall) == 0 {
		return 0, 0
	}
	f1Score := 2 * (precision * recall) / (precision + recall)
	return f1Score * 100, recall * 100
}

// get prediction as max prob in row
func Prediction(vs []float64) float64 {
	if vs[0] < 0.5 {
		return 0.0
	} else {
		return 1.0
	}
}

func StartEvaluation(Xv, Yv *mat.Dense, context actor.Context) {
	con := Config{
		Epochs:    25,
		Eta:       0.3,
		BatchSize: 32,
	}
	_, cols := Xv.Dims()
	arch := []int{cols, 15, 8, 1}
	n := New(con, arch...)

	aggregationActor, _ := context.RequestFuture(context.Parent(), &messages.GetAggregationActor{}, 5*time.Second).Result()
	globalWeightsResult, _ := context.RequestFuture(aggregationActor.(*actor.PID), &messages.GetGlobalWeights{}, 20*time.Second).Result()
	globalWeights := globalWeightsResult.(*messages.GlobalWeights)
	n.ConvertFromGlobalWeights(globalWeights)

	f1Score, recall := n.Evaluate(Xv, Yv)
	fmt.Printf("f1_score = %0.01f%%\n", f1Score)
	fmt.Printf("recall = %0.01f%%\n", recall)
}
