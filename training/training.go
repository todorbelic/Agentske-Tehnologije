package training

import (
	"fmt"
	"github.com/asynkron/protoactor-go/actor"
	"gonum.org/v1/gonum/mat"
)

func (n *MLP) Train(x, y *mat.Dense, context actor.Context) {

	r, cx := x.Dims()
	_, cy := y.Dims()

	b := n.config.BatchSize

	for e := 1; e < n.config.Epochs+1; e++ {

		for i := 0; i < r; i += b {
			k := i + b
			if k > r {
				k = r
			}
			_x := x.Slice(i, k, 0, cx)
			_y := y.Slice(i, k, 0, cy)

			n.Backward(_x, _y, context)
		}
	}
}

func StartTraining(X, Y, Xv, Yv *mat.Dense, context actor.Context) {
	con := Config{
		Epochs:    25,
		Eta:       0.3,
		BatchSize: 32,
	}
	_, cols := X.Dims()
	arch := []int{cols, 15, 8, 1}
	n := New(con, arch...)
	//n.WriteWeightsToFile("./../weights.json")
	n.Train(X, Y, context)
	f1Score, recall := n.Evaluate(Xv, Yv)
	fmt.Printf("f1_score = %0.01f%%\n", f1Score)
	fmt.Printf("recall = %0.01f%%\n", recall)
}
