package main

import (
	"fmt"
	"log"

	messages "agentske/proto"

	"agentske/preprocessing"

	console "github.com/asynkron/goconsole"
	"github.com/asynkron/protoactor-go/actor"
	"github.com/asynkron/protoactor-go/remote"
)

type PreprocessingActor struct {
}

func ConvertToProtoData(data preprocessing.Data) (*messages.Data, error) {
	protoData := &messages.Data{}

	// Assign Labels
	protoData.Labels = ConvertToInt32Slice(data.Labels)

	// Convert Histograms
	for _, hist := range data.Histograms {
		protoHist := &messages.Histogram{}
		protoHist.Values = hist

		protoData.Histograms = append(protoData.Histograms, protoHist)
	}

	return protoData, nil
}

func ConvertToInt32Slice(labels []int) []int32 {
	result := make([]int32, len(labels))
	for i, val := range labels {
		result[i] = int32(val)
	}
	return result
}

func (state *PreprocessingActor) Receive(context actor.Context) {
	switch msg := context.Message().(type) {
	case *messages.ActivatePreproc:
		log.Println("Preprocessing Actor started:", context.Self().String())
		train, val := preprocessing.PreprocessImages()
		fmt.Println(len(train.Histograms))
		fmt.Println(len(val.Histograms))
		trainProto, _ := ConvertToProtoData(train)
		valProto, _ := ConvertToProtoData(val)
		message := &messages.DataSets{Training: trainProto, Validation: valProto}
		context.Request(msg.Sender, message)
	}
}

func main() {
	actorSystem := actor.NewActorSystem()
	cfg := remote.Configure("127.0.0.1", 8100)
	r := remote.NewRemote(actorSystem, cfg)
	r.Start()

	// register a name for our local actor so that it can be spawned remotely
	r.Register("preproc", actor.PropsFromProducer(func() actor.Actor { return &PreprocessingActor{} }))
	console.ReadLine()
}
