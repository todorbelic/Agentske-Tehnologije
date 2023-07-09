package preprocessing

import (
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"

	"github.com/nfnt/resize"
)

type Params struct {
	Radius    uint8
	Neighbors uint8
	GridX     uint8
	GridY     uint8
}

var lbphParams = Params{
	Radius:    1,
	Neighbors: 8,
	GridX:     5,
	GridY:     5,
}

type Data struct {
	Labels     []float64
	Histograms [][]float64
}

func readImages(imgLocation string, label float64) ([]image.Image, []float64, error) {
	var imgPaths []string
	err := filepath.Walk(imgLocation, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			imgPaths = append(imgPaths, path)
		}
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	var images []image.Image
	var labels []float64
	for _, imgPath := range imgPaths {
		file, err := os.Open(imgPath)
		if err != nil {
			return nil, nil, err
		}
		defer file.Close()

		img, _, err := image.Decode(file)
		if err != nil {
			return nil, nil, err
		}
		resized_image := resize.Resize(150, 150, img, resize.Lanczos3)

		images = append(images, resized_image)
		labels = append(labels, label)
	}

	return images, labels, nil
}

// getBinaryString function used to get a binary value as a string based on a threshold.
// Return "1" if the value is equal or higher than the threshold or "0" otherwise.
func getBinaryString(value, threshold int) string {
	if value >= threshold {
		return "1"
	} else {
		return "0"
	}
}

func LBPHistograms(images []image.Image, labels []float64) (*Data, error) {

	// Check if the slices are not empty.
	if len(images) == 0 || len(labels) == 0 {
		return nil, errors.New("At least one of the slices is empty")
	}

	// Check if the images and labels slices have the same size.
	if len(images) != len(labels) {
		return nil, errors.New("The slices have different sizes")
	}

	// Calculates the LBP operation and gets the histograms for each image.
	var histograms [][]float64
	for index := 0; index < len(images); index++ {
		// Calculate the LBP operation for the current image.
		pixels, err := CalculateLBP(images[index], lbphParams.Radius, lbphParams.Neighbors)
		if err != nil {
			return nil, err
		}

		// Get the histogram from the current image.
		hist, err := CalculateHistograms(pixels, lbphParams.GridX, lbphParams.GridY, 64)
		if err != nil {
			return nil, err
		}

		// Store the histogram in the 'matrix' (slice of slice).
		histograms = append(histograms, hist)
	}

	// Store the current data that we are working on.
	data := &Data{
		Labels:     labels,
		Histograms: histograms,
	}

	// Everything is ok, return nil.
	return data, nil
}

// GetImageSize function is used to get the width and height from an image.
// If the image is nil it will return 0 width and 0 height
func GetImageSize(img image.Image) (int, int) {
	if img == nil {
		return 0, 0
	}
	// Get the image bounds
	bounds := img.Bounds()
	// Return the width and height
	return bounds.Max.X, bounds.Max.Y
}

// GetPixels function returns a 'matrix' ([][]uint8) containing all pixels from the image passed by parameter.
func GetPixels(img image.Image) [][]uint8 {
	var pixels [][]uint8

	// Check if the image is nil
	if img == nil {
		return pixels
	}

	// Get the image size
	width, height := GetImageSize(img)

	// For each pixel in the image (x, y) convert it to grayscale and store it in the 'matrix'
	for x := 0; x < width; x++ {
		var row []uint8
		for y := 0; y < height; y++ {
			// Get the RGB from the current pixel
			r, g, b, _ := img.At(x, y).RGBA()

			// Convert the RGB to Grayscale (red*30% + green*59% + blue*11%)
			// https://en.wikipedia.org/wiki/Grayscale#Luma_coding_in_video_systems
			pixel := (float32(r) * 0.299) + (float32(g) * 0.587) + (float32(b) * 0.114)

			// Convert the pixel from uin64 to uint8 (0-255) and append it to the slice
			row = append(row, uint8(pixel))
		}
		// Append the row (slice) to the pixels 'matrix'
		pixels = append(pixels, row)
	}

	// Return all pixels
	return pixels
}

// Calculate function calculates the LBP based on the radius and neighbors passed by parameter.
// The radius and neighbors parameters are not in use.
func CalculateLBP(img image.Image, radius, neighbors uint8) ([][]uint64, error) {

	var lbpPixels [][]uint64
	// Check the parameters
	if img == nil {
		return lbpPixels, errors.New("The image passed to the ApplyLBP function is nil")
	}
	if radius <= 0 {
		return lbpPixels, errors.New("Invalid radius parameter passed to the ApplyLBP function")
	}
	if neighbors <= 0 {
		return lbpPixels, errors.New("Invalid neighbors parameter passed to the ApplyLBP function")
	}

	// Get the pixels 'matrix' ([][]uint8)
	pixels := GetPixels(img)

	// Get the image size (width and height)
	width, height := GetImageSize(img)

	// For each pixel in the image
	for x := 1; x < width-1; x++ {
		var currentRow []uint64
		for y := 1; y < height-1; y++ {

			// Get the current pixel as the threshold
			threshold := int(pixels[x][y])

			binaryResult := ""
			// Window based on the radius (3x3)
			for tempX := x - 1; tempX <= x+1; tempX++ {
				for tempY := y - 1; tempY <= y+1; tempY++ {
					// Get the binary for all pixels around the threshold
					if tempX != x || tempY != y {
						binaryResult += getBinaryString(int(pixels[tempX][tempY]), threshold)
					}
				}
			}

			// Convert the binary string to a decimal integer
			dec, err := strconv.ParseUint(binaryResult, 2, 64)
			if err != nil {
				return lbpPixels, errors.New("Error converting binary to uint in the ApplyLBP function")
			} else {
				// Append the decimal do the result slice
				// ParseUint returns a uint64 so we need to convert it to uint8
				currentRow = append(currentRow, uint64(dec))
			}
		}
		// Append the slice to the 'matrix'
		lbpPixels = append(lbpPixels, currentRow)
	}
	return lbpPixels, nil
}

// Calculate function generates a histogram based on the 'matrix' passed by parameter.
func CalculateHistograms(pixels [][]uint64, gridX, gridY uint8, bins_per_sub_images int) ([]float64, error) {
	var hist []float64

	// Check the pixels 'matrix'
	if len(pixels) == 0 {
		return hist, errors.New("The pixels slice passed to the GetHistogram function is empty")
	}

	// Get the 'matrix' dimensions
	rows := len(pixels)
	cols := len(pixels[0])

	// Check the grid (X and Y)
	if gridX <= 0 || int(gridX) >= cols {
		return hist, errors.New("Invalid grid X passed to the GetHistogram function")
	}
	if gridY <= 0 || int(gridX) >= rows {
		return hist, errors.New("Invalid grid Y passed to the GetHistogram function")
	}

	// Get the size (width and height) of each region
	gridWidth := cols / int(gridX)
	gridHeight := rows / int(gridY)

	// Calculates the histogram of each grid
	for gX := 0; gX < int(gridX); gX++ {
		for gY := 0; gY < int(gridY); gY++ {
			// Create a slice with empty 256 positions
			regionHistogram := make([]float64, bins_per_sub_images)

			// Define the start and end positions for the following loop
			startPosX := gX * gridWidth
			startPosY := gY * gridHeight
			endPosX := (gX + 1) * gridWidth
			endPosY := (gY + 1) * gridHeight

			// Make sure that no pixel has been leave at the end
			if gX == int(gridX)-1 {
				endPosX = cols
			}
			if gY == int(gridY)-1 {
				endPosY = rows
			}

			// Creates the histogram for the current region
			for x := startPosX; x < endPosX; x++ {
				for y := startPosY; y < endPosY; y++ {
					// Make sure we are trying to access a valid position
					if x < len(pixels) {
						if y < len(pixels[x]) {
							if int(pixels[x][y]) < len(regionHistogram) {
								regionHistogram[pixels[x][y]] += 1
							}
						}
					}
				}
			}
			// Concatenate two slices
			hist = append(hist, regionHistogram...)
		}
	}

	return hist, nil
}

func shuffleData(data Data, seed int) Data {
	rand.Seed(int64(seed))
	shuffledData := Data{
		Labels:     make([]float64, len(data.Labels)),
		Histograms: make([][]float64, len(data.Histograms)),
	}

	perm := rand.Perm(len(data.Histograms))
	for i, j := range perm {
		shuffledData.Labels[i] = data.Labels[j]
		shuffledData.Histograms[i] = data.Histograms[j]
	}

	return shuffledData
}

func splitData(data Data, splitRatio float64, seed int) (Data, Data) {
	// Shuffle the data first
	shuffledData := shuffleData(data, seed)

	numSamples := len(shuffledData.Histograms)
	numTrain := int(float64(numSamples) * splitRatio)

	trainData := Data{
		Labels:     make([]float64, numTrain),
		Histograms: make([][]float64, numTrain),
	}

	valData := Data{
		Labels:     make([]float64, numSamples-numTrain),
		Histograms: make([][]float64, numSamples-numTrain),
	}

	for i := 0; i < numTrain; i++ {
		trainData.Labels[i] = shuffledData.Labels[i]
		trainData.Histograms[i] = shuffledData.Histograms[i]
	}

	for i := numTrain; i < numSamples; i++ {
		valData.Labels[i-numTrain] = shuffledData.Labels[i]
		valData.Histograms[i-numTrain] = shuffledData.Histograms[i]
	}

	return trainData, valData
}

func PreprocessImages() (Data, Data) {
	normalLungsImgLocation := "data\\normal_read"
	normalLungsImages, normalLabels, err := readImages(normalLungsImgLocation, 0.0)
	if err != nil {
		fmt.Println("Error loading normal lung images:", err)
	}

	covidLungsImgLocation := "data\\covid_read"
	covidLungsImages, covidLabels, err := readImages(covidLungsImgLocation, 1.0)
	if err != nil {
		fmt.Println("Error loading covid lung images: ", err)
	}

	allImages, allLabels := append(normalLungsImages, covidLungsImages...), append(normalLabels, covidLabels...)

	preprocessedAllImages, err := LBPHistograms(allImages, allLabels)
	if err != nil {
		fmt.Println("Error preprocessing images: ", err)
	}

	trainData, validationData := splitData(*preprocessedAllImages, 0.7, 42)

	fmt.Println(len(trainData.Histograms))
	fmt.Println(len(validationData.Histograms))

	return trainData, validationData
}

//func main() {
//	normalLungsImgLocation := "data\\normal"
//	normalLungsImages, normalLabels, err := readImages(normalLungsImgLocation, 0.0)
//	if err != nil {
//		fmt.Println("Error loading normal lung images:", err)
//		return
//	}
//
//	covidLungsImgLocation := "data\\covid"
//	covidLungsImages, covidLabels, err := readImages(covidLungsImgLocation, 1.0)
//	if err != nil {
//		fmt.Println("Error loading covid lung images: ", err)
//		return
//	}
//
//	allImages, allLabels := append(normalLungsImages, covidLungsImages...), append(normalLabels, covidLabels...)
//
//	preprocessedAllImages, err := LBPHistograms(allImages, allLabels)
//	if err != nil {
//		fmt.Println("Error preprocessing images: ", err)
//		return
//	}
//
//	trainData, validationData := splitData(*preprocessedAllImages, 0.7, 42)
//
//	fmt.Println(len(trainData.Histograms[0]))
//	fmt.Println(len(validationData.Histograms))
//
//	con := nn.Config{
//		Epochs:    25,
//		Eta:       0.3,
//		BatchSize: 32,
//	}
//
//	arch := []int{len(trainData.Histograms[0]), 15, 8, 1}
//	n := nn.New(con, arch...)
//
//	rows, cols := len(trainData.Histograms), len(trainData.Histograms[0])
//	data := make([]float64, rows*cols)
//	for i, row := range trainData.Histograms {
//		copy(data[i*cols:(i+1)*cols], row)
//	}
//	dense := mat.NewDense(rows, cols, data)
//	denseY := mat.NewDense(rows, 1, trainData.Labels)
//
//	dataev := make([]float64, len(validationData.Histograms)*cols)
//	for i, row := range validationData.Histograms {
//		copy(dataev[i*cols:(i+1)*cols], row)
//	}
//	denseev := mat.NewDense(len(validationData.Histograms), cols, dataev)
//	denseYev := mat.NewDense(len(validationData.Histograms), 1, validationData.Labels)
//
//	n.Train(dense, denseY)
//
//	accuracy := n.Evaluate(denseev, denseYev)
//
//	fmt.Printf("accuracy = %0.1f%%\n", accuracy)
//
//}
