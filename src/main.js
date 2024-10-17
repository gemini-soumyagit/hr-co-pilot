
export default async ({ req, res, log, error }) => {
  if(req.method.get){

    return res.send("welcome to era of AI !")
  }
  if(req.method.post){

    return res.json({
      'sendData':req.body,
      'googleAPiKey': process.env.GOOGLE_GEMINI_API_KEY
    })
  }

};
